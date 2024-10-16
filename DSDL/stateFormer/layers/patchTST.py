__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.layers import PatchTST_backbone, SeriesDecomp, CategoricalEmbedding

class PatchTST(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, dimension_keys:Optional[int]=None, dimension_values:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, positional_encoding:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        print('PatchTST model is being used')
        print('Configs:', configs)
        # load parameters
        input_channel_num = len(configs['numerical_features'])
        context_window = configs['state_space_embedding']
        target_window = configs['prediction_horizon']
        
        n_layers = configs['transformer_layers']
        n_heads = configs['attention_heads']
        d_model = configs['hidden_state_dim']
        d_ff = configs['feedforward_dim']
        dropout = configs['dropout']
        fc_dropout = configs['dropout']
        head_dropout = configs['dropout']
        
        individual = False
        patch_len = int(configs['patch_len'])
        stride = int(configs['patch_len'] * configs['stride'])
        padding_patch = 'end'
        
        revin = False
        affine = False
        subtract_last = 0
        
        decomposition = configs['decomposition']
        kernel_size = configs['kernel_size']
        
        # New parameters for categorical embeddings
        self.cat_dims = configs['categorical_cardinalities'] if 'categorical_cardinalities' in configs else []
        self.cat_emb_dim = configs['categorical_embedding_dim'] if 'categorical_embedding_dim' in configs else 8
        
        # Create embedding layers for categorical features
        if self.cat_dims:
            self.categorical_embedding = CategoricalEmbedding(self.cat_dims, self.cat_emb_dim)
            input_channel_num += len(self.cat_dims) * self.cat_emb_dim
        
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = SeriesDecomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=input_channel_num, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, dimension_keys=dimension_keys, d_v=dimension_values, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=positional_encoding, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=input_channel_num, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, dimension_keys=dimension_keys, d_v=dimension_values, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=positional_encoding, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=input_channel_num, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                    max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                    n_heads=n_heads, dimension_keys=dimension_keys, d_v=dimension_values, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                    dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                    attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                    pe=positional_encoding, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                    pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                    subtract_last=subtract_last, verbose=verbose, **kwargs)
    
    
    def forward(self, x):           # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            # our input is already in the correct shape B,C,I
            # x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return x