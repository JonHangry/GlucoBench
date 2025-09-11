import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from typing import Optional

import numpy as np
# from utils.masking import TriangularCausalMask, ProbMask
from utils.rotation import RoPE1d
from einops import rearrange


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 8, d_k=None, d_v=None, res_attention=False,
                 attn_dropout: float = 0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout,
                                                  res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        # Q: [batch_size * channels x patch_num_total x dim_model]
        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,
                                                                         2)  # q_s    : [bs x n_heads x channels x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3,
                                                                       1)  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)  # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev,
                                                              key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1,
                                                          self.n_heads * self.d_v)  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q: Tensor, k: Tensor, v: Tensor, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale  # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:  # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


# Informer Prob
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


# Crossformer-TSA
class TwoStageAttentionLayer(nn.Module):
    def __init__(self, batch_size: int = 32, channels: int = 21, patch_num: int = 41, factor: int = 10,
                 d_model: int = 128, n_heads: int = 8, d_ff: int = 256,
                 attn_dropout: float = 0., dropout: float = 0.2, norm: str = 'BatchNorm',
                 activation=None, transpose=None, bias: bool = True):
        super(TwoStageAttentionLayer, self).__init__()
        self.batch_size = batch_size
        self.channels = channels
        # Stage-1: Cross-Time-Attention
        self.time_attention = MultiheadAttention(d_model=d_model, n_heads=n_heads,
                                                 attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=False)
        # Stage-2: Cross-Dimension-Attention
        self.dim_sender = MultiheadAttention(d_model=d_model, n_heads=n_heads,
                                             attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=False)
        self.dim_receiver = MultiheadAttention(d_model=d_model, n_heads=n_heads,
                                               attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=False)
        self.router = nn.Parameter(torch.randn(patch_num, factor, d_model))
        # Add & Norm
        self.dropout_attn1 = nn.Dropout(dropout)
        self.dropout_ffn1 = nn.Dropout(dropout)
        self.dropout_attn2 = nn.Dropout(dropout)
        self.dropout_ffn2 = nn.Dropout(dropout)
        # feedforward neural network
        self.ff1 = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                 activation,
                                 nn.Linear(d_ff, d_model, bias=bias))
        self.ff2 = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                 activation,
                                 nn.Linear(d_ff, d_model, bias=bias))
        if "Batch" in norm:
            self.norm_attn1 = nn.Sequential(transpose(1, 2), nn.BatchNorm1d(d_model), transpose(1, 2))
            self.norm_ffn1 = nn.Sequential(transpose(1, 2), nn.BatchNorm1d(d_model), transpose(1, 2))
            self.norm_attn2 = nn.Sequential(transpose(1, 2), nn.BatchNorm1d(d_model), transpose(1, 2))
            self.norm_ffn2 = nn.Sequential(transpose(1, 2), nn.BatchNorm1d(d_model), transpose(1, 2))
        else:
            self.norm_attn1 = nn.LayerNorm(d_model)
            self.norm_ffn1 = nn.LayerNorm(d_model)
            self.norm_attn2 = nn.LayerNorm(d_model)
            self.norm_ffn2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Stage-1: CTA
        # x: [batch_size * channels x patch_num x d_model]
        result, _ = self.time_attention(x)
        # Add & Norm
        output = x + self.dropout_attn1(result)
        output = self.norm_attn1(output)
        output = output + self.dropout_ffn1(self.ff1(output))
        output = self.norm_ffn1(output)

        # Stage-2: CDA
        # output: [batch_size * channels x patch_num x d_model] � dim_send: [patch_num x channels x d_model]
        shape = output.shape
        # dim_send: [patch_num x channels x d_model] � [batch_size * patch_num x channels x d_model]
        dim_send = torch.reshape(output, (-1, self.channels, output.shape[1], output.shape[2])).permute(0, 2, 1, 3)
        dim_send = torch.reshape(dim_send, (-1, self.channels, output.shape[2]))
        batch_router = self.router.repeat(self.batch_size, 1, 1)
        dim_buffer, _ = self.dim_sender(batch_router, dim_send, dim_send)
        dim_receive, _ = self.dim_receiver(dim_send, dim_buffer, dim_buffer)
        # Add & Norm
        output = dim_send + self.dropout_attn2(dim_receive)
        output = self.norm_attn2(output)
        output = output + self.dropout_ffn2(self.ff2(output))
        output = self.norm_ffn2(output)
        # dim_enc: [batch_size * patch_num x channels x d_model] � [batch_size x channels x patch_num x d_model]
        output = output.reshape(self.batch_size, -1, output.shape[1], output.shape[2]).permute(0, 2, 1, 3)
        output = output.reshape(output.shape[0] * output.shape[1], output.shape[2], output.shape[3])
        return output


class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """

    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k of corr
        top_k = int(self.factor * math.log(length))
        # corr: [bs x nh x emb x seq_len] -> [bs x emb x seq_len] -> [bs x seq_len]
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights = torch.topk(mean_value, top_k, dim=-1)[0]
        delay = torch.topk(mean_value, top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # find top k
        top_k = int(self.factor * math.log(length))
        weights = torch.topk(corr, top_k, dim=-1)[0]
        delay = torch.topk(corr, top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask=None):
        # Q,K,V: [bs x seq_len x n_heads x embed]
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class LinearAttention(nn.Module):
    def __init__(self, attention_dropout=0.1, output_attention=False) -> None:
        super().__init__()
        self.drop = nn.Dropout(attention_dropout)
        self.elu = nn.ELU()

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        queries = self.elu(queries) + 1
        keys = self.elu(keys) + 1

        y = 1 / (queries @ keys.sum(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)  # B H L S
        kv = self.drop((keys.transpose(-2, -1) * (L ** -0.5)) @ values)  # B H E D
        x = queries @ kv * y  # B L H E
        return x, None


class RouterAttention(nn.Module):
    """
    Router Attention layer.
    """

    def __init__(self, router_num, d_model, n_heads=8,
                 rotary=True, residual=True, gate=True,
                 attention_dropout=0.1, output_attention=False):
        super(RouterAttention, self).__init__()
        self.rotary = rotary
        self.residual = residual
        self.gate = gate
        self.n_heads = n_heads

        d_keys = d_model // n_heads
        d_values = d_model // n_heads

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)

        if self.residual:
            self.skip_projection = nn.Linear(d_values * n_heads, d_model)
        if self.rotary:
            self.rope = RoPE1d(feature_dim=router_num, reverse=True)

        self.drop1 = nn.Dropout(attention_dropout)
        self.drop2 = nn.Dropout(attention_dropout)

        # self.router_proj = nn.AdaptiveAvgPool1d(output_size=agent_num)
        # self.router_proj = nn.Conv1d(seq_len, router_num, 3, 1, 1)
        self.router_proj = nn.Parameter(torch.randn(router_num, d_model))
        if self.gate:
            self.z_projection = nn.Linear(d_model, d_model)
            self.act = nn.SiLU()

    def forward(self, x, *args, **kwargs):
        # [B, L, D]
        if self.gate:
            z = self.act(self.z_projection(x))

        queries = self.query_projection(x)
        keys = self.key_projection(x)

        # [B, L, D] -> [B, a, D]
        # routers = self.router_proj(queries.permute(0, 2, 1)).permute(0, 2, 1)
        # routers = self.router_proj(x)
        routers = self.router_proj.repeat(x.shape[0], 1, 1)

        q = rearrange(queries, "B L (H E) -> B H L E", H=self.n_heads)
        k = rearrange(keys, "B S (H E) -> B H S E", H=self.n_heads)
        v = rearrange(x, "B S (H D) -> B H S D", H=self.n_heads)
        r = rearrange(routers, "B r (H E) -> B H r E", H=self.n_heads)

        scale = 1. / math.sqrt(k.shape[-1])
        router_scores = torch.einsum("BHrE, BHSE -> BHrS", r, k)
        router_A = torch.softmax(scale * router_scores, dim=-1)
        if self.rotary:
            router_A = self.rope(router_A.transpose(-1, -2)).transpose(-1, -2)
        router_A = self.drop1(router_A)
        router_V = torch.einsum("BHrS, BHSD -> BHrD", router_A, v)

        scale = 1. / math.sqrt(q.shape[-1])
        q_scores = torch.einsum("BHLE, BHrE -> BHLr", q, r)
        q_A = torch.softmax(scale * q_scores, dim=-1)
        if self.rotary:
            q_A = self.rope(q_A)
        q_A = self.drop2(q_A)
        V = torch.einsum("BHLr, BHrD -> BHLD", q_A, router_V)
        V = rearrange(V, "B H L D -> B L (H D)")
        if self.residual:
            V = V + self.skip_projection(x)
        if self.gate:
            V = V * z
        return V

class QKV(nn.Module):
    """
    Router Attention layer.
    """

    def __init__(self, d_model, n_heads=8):
        super(QKV, self).__init__()
        self.n_heads = n_heads

        d_keys = d_model // n_heads
        d_values = d_model // n_heads

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)


    def forward(self, x, *args, **kwargs):
        # [B, L, D]

        queries = self.query_projection(x)
        keys = self.key_projection(x)

        # [B, L, D] -> [B, a, D]
        # routers = self.router_proj(queries.permute(0, 2, 1)).permute(0, 2, 1)
        # routers = self.router_proj(x)

        q = rearrange(queries, "B L (H E) -> B H L E", H=self.n_heads)
        k = rearrange(keys, "B S (H E) -> B H S E", H=self.n_heads)
        v = rearrange(x, "B S (H D) -> B H S D", H=self.n_heads)

        return  q,k,v


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def _update_cos_sin_cache(self, x, seq_len):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached, self.sin_cached

    def forward(self, q, k):
        cos, sin = self._update_cos_sin_cache(q, q.shape[-2])
        return (
            (q * cos) + (self._rotate_half(q) * sin),
            (k * cos) + (self._rotate_half(k) * sin),
        )

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)

    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(query.device)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias

    if attn_mask is not None:
        attn_mask_bias = torch.zeros_like(attn_weight)
        if attn_mask.dtype == torch.bool:
            attn_mask_bias.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_mask_bias += attn_mask
        attn_weight += attn_mask_bias

    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout_p=0.0, resid_dropout_p=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

    def forward(self, x, *args, **kwargs):
        batch_size, seq_len, _ = x.shape
        key_padding_mask = None
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = self.rotary(q, k)

        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attn_mask = attn_mask.expand(-1, self.n_heads, seq_len, -1)  # [batch, n_heads, q_len, k_len]
        else:
            attn_mask = None

        attn_output = scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p,
            is_causal=True
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.resid_dropout(self.out_proj(attn_output))


class MultiHeadCrossAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout_p=0.0, resid_dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        batch_size, q_len, _ = query.shape
        _, seq_len, _ = key.shape

        q = self.q_proj(query).view(batch_size, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = self.rotary(q, k)

        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, self.n_heads, q_len, -1)
        else:
            attn_mask = None

        is_causal_flag = self.training

        attn_output = scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p,
            is_causal=is_causal_flag
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        return self.resid_dropout(self.out_proj(attn_output))