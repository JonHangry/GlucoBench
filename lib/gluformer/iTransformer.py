import os
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
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

class iTrans(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(iTrans, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.len_pred = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # Train variance
        self.var = Variance(configs.d_model, configs.r_drop, configs.seq_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        var_out = self.var(enc_out)
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, var_out


    def forward(self, x_id, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, var_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], var_out
        else:
            return dec_out[:, -self.pred_len:, :], var_out  # [B, L, D]

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