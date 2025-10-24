__all__ = ['PatchTST']

import os
import sys
from tqdm import tqdm
from .variance import *

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp
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

class PTST(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        
        
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
    
    
    def forecast(self, x):           # x: [Batch, Input length, Channel]
        var_out = self.var(x)
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return x,var_out

    def forward(self, x_enc):
        dec_out, var_out = self.forecast(x_enc)

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