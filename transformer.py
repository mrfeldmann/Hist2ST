import torch
import numpy as np
import pandas as pd
import scanpy as sc
import torch.nn.functional as F
from easydl import *
from anndata import AnnData
from torch import nn, einsum
from scipy.stats import pearsonr
from torch.autograd import Function
from torch.autograd.variable import *
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        # heads = number of attention pooling heads in one layer
        #dim = number of filters of input patch ("we set both input and hidden dimension = 1024") 
        # I dont know why they chose dim_head = 64 or what dim_head actually is.
        # I think dim_head are 64 attention pooling modules in a single multi-head attention layer
        inner_dim = dim_head *  heads 
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        # in comparison to what's done in fastai or torch
        # here the three projections of the input patches, i.e.,
        # q,k,v are are embedded in a higher dimension (=inner_dim)
        # than the input number of filters (64 * 8 > 1024).
        # inner_dim is multiplied by 3 as to create 3 projections which
        # are created by chunkifying this linear layer later, just as
        # it is done in fastai22p2 notebook 27
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
    # @get_local('attn')
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        #create the three projections of the input patch
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # reshape the projections, so that each head 
        # will see a subset of the input channels:
        # the channels (= h x d) will be equally devided among
        # the heads (h) --> 
        # inner_dim = 64 x 8 divided by h=8 heads -> d = 64 filters analyzed per attention pooling head.
        # thus, the "dimension" of each head would be 64 which is probably the dim_head parameter
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        # calculate dot products of the q and k projections to 
        # quantify similarity.
        # since values from each filter will be added up, this will increase 
        # the scale. thus, the values are rescaled to create more 
        # sensible values
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        # calculate softmax -> create weights between 0 and 1.
        # also, high weights will get even bigger proportionally,
        # which will lead to only a relatively small subset of pixels 
        # being paid attention to
        attn = self.attend(dots)
        # calculate new representation of the patch based on the attention 
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class attn_block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.attn=PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))
        self.ff=PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x
