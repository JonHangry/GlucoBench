import os
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
### Mixer lib(maybe useless)
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
###########
from .embed import *
from .attention import *
from .encoder import *
from .decoder import *
from .variance import *

############################################
# Added for GluNet package
############################################
import optuna
import darts
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from lib.gluformer.utils.training import ExpLikeliLoss, \
                                         EarlyStop, \
                                         modify_collate, \
                                         adjust_learning_rate
from utils.darts_dataset import SamplingDatasetDual
############################################

### Mixer class(maybe useless)
class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence

        if configs.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError('decompsition is error')

        if configs.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list

# HonBan
class Gluformer(nn.Module):
    def __init__(self, configs):  # 暂时保留，之后说不定用def __init__(self, configs):
        super(Gluformer, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.len_pred = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in
        self.use_future_temporal_feature = configs.use_future_temporal_feature

        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        self.layer = configs.e_layers

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)

                self.out_res_layers = torch.nn.ModuleList([
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ])

                self.regression_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )

        # 暂时没有添加以下功能的想法
        # if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
        #     if self.channel_independence == 1:
        #         self.projection_layer = nn.Linear(
        #             configs.d_model, 1, bias=True)
        #     else:
        #         self.projection_layer = nn.Linear(
        #             configs.d_model, configs.c_out, bias=True)
        # if self.task_name == 'classification':
        #     self.act = F.gelu
        #     self.dropout = nn.Dropout(configs.dropout)
        #     self.projection = nn.Linear(
        #         configs.d_model * configs.seq_len, configs.num_class)

        # Embedding   former原版的embed，还没想好是否使用
        # note: d_model // 2 == 0
        # Train variance
        self.var = Variance(configs.d_model, configs.r_drop, configs.seq_len)
        # Embedding为止是原版残余


    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out


    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)


    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_future_temporal_feature:
            if self.channel_independence == 1:
                B, T, N = x_enc.size()
                x_mark_dec = x_mark_dec.repeat(N, 1, 1)
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
            else:
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)

        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
                # 这里存疑
                if i == 0:
                    var_out = self.var(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)
                if i == 0:
                    var_out = self.var(enc_out)

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out, var_out


    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                if self.use_future_temporal_feature:
                    dec_out = dec_out + self.x_mark_dec
                    dec_out = self.projection_layer(dec_out)
                else:
                    dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

        # 暂不使用
        # def classification(self, x_enc, x_mark_enc):
        #     x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)
        #     x_list = x_enc
        #
        #     # embedding
        #     enc_out_list = []
        #     for x in x_list:
        #         enc_out = self.enc_embedding(x, None)  # [B,T,C]
        #         enc_out_list.append(enc_out)
        #
        #     # MultiScale-CrissCrossAttention  as encoder for past
        #     for i in range(self.layer):
        #         enc_out_list = self.pdm_blocks[i](enc_out_list)
        #
        #     enc_out = enc_out_list[0]
        #     # Output
        #     # the output transformer encoder/decoder embeddings don't include non-linearity
        #     output = self.act(enc_out)
        #     output = self.dropout(output)
        #     # zero-out padding embeddings
        #     output = output * x_mark_enc.unsqueeze(-1)
        #     # (batch_size, seq_length * d_model)
        #     output = output.reshape(output.shape[0], -1)
        #     output = self.projection(output)  # (batch_size, num_classes)
        #     return output
        #
        # def anomaly_detection(self, x_enc):
        #     B, T, N = x_enc.size()
        #     x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)
        #
        #     x_list = []
        #
        #     for i, x in zip(range(len(x_enc)), x_enc, ):
        #         B, T, N = x.size()
        #         x = self.normalize_layers[i](x, 'norm')
        #         if self.channel_independence == 1:
        #             x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        #         x_list.append(x)
        #
        #     # embedding
        #     enc_out_list = []
        #     for x in x_list:
        #         enc_out = self.enc_embedding(x, None)  # [B,T,C]
        #         enc_out_list.append(enc_out)
        #
        #     # MultiScale-CrissCrossAttention  as encoder for past
        #     for i in range(self.layer):
        #         enc_out_list = self.pdm_blocks[i](enc_out_list)
        #
        #     dec_out = self.projection_layer(enc_out_list[0])
        #     dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()
        #
        #     dec_out = self.normalize_layers[0](dec_out, 'denorm')
        #     return dec_out
        #
        # def imputation(self, x_enc, x_mark_enc, mask):
        #     means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        #     means = means.unsqueeze(1).detach()
        #     x_enc = x_enc - means
        #     x_enc = x_enc.masked_fill(mask == 0, 0)
        #     stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
        #                        torch.sum(mask == 1, dim=1) + 1e-5)
        #     stdev = stdev.unsqueeze(1).detach()
        #     x_enc /= stdev
        #
        #     B, T, N = x_enc.size()
        #     x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
        #
        #     x_list = []
        #     x_mark_list = []
        #     if x_mark_enc is not None:
        #         for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
        #             B, T, N = x.size()
        #             if self.channel_independence == 1:
        #                 x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        #             x_list.append(x)
        #             x_mark = x_mark.repeat(N, 1, 1)
        #             x_mark_list.append(x_mark)
        #     else:
        #         for i, x in zip(range(len(x_enc)), x_enc, ):
        #             B, T, N = x.size()
        #             if self.channel_independence == 1:
        #                 x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        #             x_list.append(x)
        #
        #     # embedding
        #     enc_out_list = []
        #     for x in x_list:
        #         enc_out = self.enc_embedding(x, None)  # [B,T,C]
        #         enc_out_list.append(enc_out)
        #
        #     # MultiScale-CrissCrossAttention  as encoder for past
        #     for i in range(self.layer):
        #         enc_out_list = self.pdm_blocks[i](enc_out_list)
        #
        #     dec_out = self.projection_layer(enc_out_list[0])
        #     dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()
        #
        #     dec_out = dec_out * \
        #               (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        #     dec_out = dec_out + \
        #               (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        #     return dec_out
        ########暂不使用

        # 主forward


    def forward(self, x_id, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, var_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out, var_out
        # if self.task_name == 'imputation':
        #     dec_out = self.imputation(x_enc, x_mark_enc, mask)
        #     return dec_out  # [B, L, D]
        # if self.task_name == 'anomaly_detection':
        #     dec_out = self.anomaly_detection(x_enc)
        #     return dec_out  # [B, L, D]
        # if self.task_name == 'classification':
        #     dec_out = self.classification(x_enc, x_mark_enc)
        #     return dec_out  # [B, N]
        else:
            raise ValueError('Other tasks implemented yet')

        ############################################
        # Added for GluNet package
        ############################################


    def fit(self,
            train_dataset: SamplingDatasetDual,
            val_dataset: SamplingDatasetDual,
            learning_rate: float = 1e-3,
            batch_size: int = 32,
            epochs: int = 100,
            num_samples: int = 100,
            device: str = 'cuda',
            model_path: str = None,
            trial: optuna.trial.Trial = None,
            logger: SummaryWriter = None, ):
        """
        Fit the model to the data, using Optuna for hyperparameter tuning.

        Parameters
        ----------
        train_dataset: SamplingDatasetPast
          Training dataset.
        val_dataset: SamplingDatasetPast
          Validation dataset.
        learning_rate: float
          Learning rate for Adam.
        batch_size: int
          Batch size.
        epochs: int
          Number of epochs.
        num_samples: int
          Number of samples for infinite mixture
        device: str
          Device to use.
        model_path: str
          Path to save the model.
        trial: optuna.trial.Trial
          Trial for hyperparameter tuning.
        logger: SummaryWriter
          Tensorboard logger for logging.
        """
        # create data loaders, optimizer, loss, and early stopping
        collate_fn_custom = modify_collate(num_samples)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   drop_last=True,
                                                   collate_fn=collate_fn_custom)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 drop_last=True,
                                                 collate_fn=collate_fn_custom)
        criterion = ExpLikeliLoss(num_samples)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(0.1, 0.9))
        scaler = torch.cuda.amp.GradScaler()
        early_stop = EarlyStop(patience=10, delta=0.001)
        self.to(device)
        # train and evaluate the model
        for epoch in range(epochs):
            train_loss = []
            for i, (past_target_series,
                    past_covariates,
                    future_covariates,
                    static_covariates,
                    future_target_series) in enumerate(train_loader):
                # zero out gradient
                optimizer.zero_grad()
                # reshape static covariates to be [batch_size, num_static_covariates]
                static_covariates = static_covariates.reshape(-1, static_covariates.shape[-1])
                # create decoder input: pad with zeros the prediction sequence
                dec_inp = torch.cat([past_target_series[:, -self.label_len:, :],
                                     torch.zeros([
                                         past_target_series.shape[0],
                                         self.len_pred,
                                         past_target_series.shape[-1]
                                     ])],
                                    dim=1)
                future_covariates = torch.cat([past_covariates[:, -self.label_len:, :],
                                               future_covariates], dim=1)
                # move to device
                dec_inp = dec_inp.to(device)
                past_target_series = past_target_series.to(device)
                past_covariates = past_covariates.to(device)
                future_covariates = future_covariates.to(device)
                static_covariates = static_covariates.to(device)
                future_target_series = future_target_series.to(device)
                # forward pass with autograd
                with torch.cuda.amp.autocast():
                    pred, logvar = self(static_covariates,
                                        past_target_series,
                                        past_covariates,
                                        dec_inp,
                                        future_covariates)
                    loss = criterion(pred, future_target_series, logvar)
                # backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # log loss
                if logger is not None:
                    logger.add_scalar('train_loss', loss.item(), epoch * len(train_loader) + i)
                train_loss.append(loss.item())
            # log loss
            if logger is not None:
                logger.add_scalar('train_loss_epoch', np.mean(train_loss), epoch)
            # evaluate the model
            val_loss = []
            with torch.no_grad():
                for i, (past_target_series,
                        past_covariates,
                        future_covariates,
                        static_covariates,
                        future_target_series) in enumerate(val_loader):
                    # reshape static covariates to be [batch_size, num_static_covariates]
                    static_covariates = static_covariates.reshape(-1, static_covariates.shape[-1])
                    # create decoder input
                    dec_inp = torch.cat([past_target_series[:, -self.label_len:, :],
                                         torch.zeros([
                                             past_target_series.shape[0],
                                             self.len_pred,
                                             past_target_series.shape[-1]
                                         ])],
                                        dim=1)
                    future_covariates = torch.cat([past_covariates[:, -self.label_len:, :],
                                                   future_covariates], dim=1)
                    # move to device
                    dec_inp = dec_inp.to(device)
                    past_target_series = past_target_series.to(device)
                    past_covariates = past_covariates.to(device)
                    future_covariates = future_covariates.to(device)
                    static_covariates = static_covariates.to(device)
                    future_target_series = future_target_series.to(device)
                    # forward pass
                    pred, logvar = self(static_covariates,
                                        past_target_series,
                                        past_covariates,
                                        dec_inp,
                                        future_covariates)
                    loss = criterion(pred, future_target_series, logvar)
                    val_loss.append(loss.item())
                    # log loss
                    if logger is not None:
                        logger.add_scalar('val_loss', loss.item(), epoch * len(val_loader) + i)
            # log loss
            logger.add_scalar('val_loss_epoch', np.mean(val_loss), epoch)
            # check early stopping
            early_stop(np.mean(val_loss), self, model_path)
            if early_stop.stop:
                break
            # check pruning
            if trial is not None:
                trial.report(np.mean(val_loss), epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
        # load best model
        if model_path is not None:
            self.load_state_dict(torch.load(model_path))


    def predict(self, test_dataset: SamplingDatasetDual,
                batch_size: int = 32,
                num_samples: int = 100,
                device: str = 'cuda',
                use_tqdm: bool = False):
        """
        Predict the future target series given the supplied samples from the dataset.

        Parameters
        ----------
        test_dataset : SamplingDatasetInferenceDual
            The dataset to use for inference.
        batch_size : int, optional
            The batch size to use for inference, by default 32
        num_samples : int, optional
            The number of samples to use for inference, by default 100

        Returns
        -------
        Predictions
            The predicted future target series in shape n x len_pred x num_samples, where
            n is total number of predictions.
        Logvar
            The logvariance of the predicted future target series in shape n x len_pred.
        """
        # define data loader
        collate_fn_custom = modify_collate(num_samples)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  drop_last=False,
                                                  collate_fn=collate_fn_custom)
        # predict
        self.train()
        # move to device
        self.to(device)
        predictions = [];
        logvars = []
        for i, (past_target_series,
                historic_future_covariates,
                future_covariates,
                static_covariates) in enumerate(tqdm(test_loader)) if use_tqdm else enumerate(test_loader):
            # reshape static covariates to be [batch_size, num_static_covariates]
            static_covariates = static_covariates.reshape(-1, static_covariates.shape[-1])
            # create decoder input
            dec_inp = torch.cat([past_target_series[:, -self.label_len:, :],
                                 torch.zeros([
                                     past_target_series.shape[0],
                                     self.len_pred,
                                     past_target_series.shape[-1]
                                 ])],
                                dim=1)
            future_covariates = torch.cat([historic_future_covariates[:, -self.label_len:, :],
                                           future_covariates], dim=1)
            # move to device
            dec_inp = dec_inp.to(device)
            past_target_series = past_target_series.to(device)
            historic_future_covariates = historic_future_covariates.to(device)
            future_covariates = future_covariates.to(device)
            static_covariates = static_covariates.to(device)
            # forward pass
            pred, logvar = self(static_covariates,
                                past_target_series,
                                historic_future_covariates,
                                dec_inp,
                                future_covariates)
            # transfer in numpy and arrange sample along last axis
            pred = pred.cpu().detach().numpy()
            logvar = logvar.cpu().detach().numpy()
            pred = pred.transpose((1, 0, 2)).reshape((pred.shape[1], -1, num_samples)).transpose((1, 0, 2))
            logvar = logvar.transpose((1, 0, 2)).reshape((logvar.shape[1], -1, num_samples)).transpose((1, 0, 2))
            predictions.append(pred)
            logvars.append(logvar)
        predictions = np.concatenate(predictions, axis=0)
        logvars = np.concatenate(logvars, axis=0)
        return predictions, logvars
  ############################################




