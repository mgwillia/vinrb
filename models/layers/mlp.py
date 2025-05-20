### code adapted from https://github.com/abhyantrika/mediainr ###

import torch 
import torch.nn as nn
from math import sqrt

from utils import get_activation, Sine

class MLPLayer(nn.Module):
    """Implements a single MLP layer.

    Args:
        dim_in (int): Dimension of input.
        dim_out (int): Dimension of output.
        activation (torch.nn.Module): Activation function. If None, defaults to
            ReLU activation.
    """
    def __init__(self, dim_in, dim_out,is_first=False,
                 use_bias=True, activation=None, **kwargs):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)
        self.activation = nn.ReLU() if activation is None else activation

        if isinstance(self.activation, Sine):
            w_std = (1 / dim_in) if self.is_first else (sqrt(self.activation.c / dim_in) / self.activation.w0)
            nn.init.uniform_(self.linear.weight, -w_std, w_std)
            if use_bias:
                nn.init.uniform_(self.linear.bias, -w_std, w_std)

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        return out
    

class NeRVMLP(nn.Module):
    def __init__(self, dim_list, act='relu', bias=True, **kwargs):
        super().__init__()
        act_fn = get_activation(act, **kwargs)
        layers = []
        for i in range(len(dim_list) - 1):
            layers.append(MLPLayer(dim_list[i], dim_list[i+1], is_first=(i==0), use_bias=bias, activation=act_fn))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MLP(nn.Module):
    """
    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        num_layers (int): Number of layers.
        use_bias (bool):
        final_activation (torch.nn.Module): Activation function.
    """
    
    def __init__(self, dim_in: int, num_layers: int, dim_hidden: int, activation=None,\
                 use_bias: bool =True,final_activation =None,**kwargs):        
        super().__init__()
        
        self.dim_in = dim_in
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.use_bias = use_bias
        self.activation = nn.ReLU() if activation is None else get_activation(activation, **kwargs)
        self.final_activation = final_activation
        self.params = kwargs
        
        self.positional_encoder = nn.Identity() #can be changed outside this class
        
        self.patch_size = self.params.get('patch_size',None)
        self.group_size = self.params.get('group_size',1)

        out_size = 3*self.group_size

        if self.patch_size is None:
            self.dim_out = out_size
        else:
            self.dim_out = (self.patch_size**2) * out_size

        layers = []
        for ind in range(self.num_layers-1):
            is_first = ind == 0
            
            layer_dim_in = self.dim_in if is_first else self.dim_hidden

            layers.append(MLPLayer(
                dim_in=layer_dim_in,
                dim_out=self.dim_hidden,
                use_bias=self.use_bias,
                is_first=is_first,
                activation=self.activation,
                **self.params
            ))

        self.final_activation = nn.Identity() if self.final_activation is None else self.final_activation
        layers.append(MLPLayer(dim_in=self.dim_hidden, dim_out=self.dim_out,
                                use_bias=self.use_bias, activation=self.final_activation))

        self.net = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        x = self.positional_encoder(x)
        out = self.net(x)

        if self.patch_size is not None:            
            out = out.view(self.group_size,out.size(0),3,self.patch_size,self.patch_size)
        
        return out