import os
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np
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

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])
        x_glb_attn = torch.reshape(x_glb_attn,
                                   (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class TimeXer(nn.Module):

    def __init__(self, configs):
        super(TimeXer, self).__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.len_pred = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = 1
        self.patch_num = int(configs.seq_len // configs.patch_len)
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in
        # Embedding
        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)

        self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                   configs.dropout)

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)
        # Train variance
        self.var = Variance(configs.d_model, configs.r_drop, configs.seq_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        en_embed, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)

        var_out = self.var(enc_out)

        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out,var_out


    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)

        var_out = self.var(enc_out)

        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out,var_out

    def forward(self, x_id, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if self.features == 'M':
                dec_out,var_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :],var_out  # [B, L, D]
            else:
                dec_out,var_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :],var_out  # [B, L, D]
        else:
            return None

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