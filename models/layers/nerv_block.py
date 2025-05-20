### code adapted from https://github.com/abhyantrika/mediainr ###

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from functools import partial
import einops

from utils import get_activation, weight_quantize_fn, assert_shape, FastTrilinearInterpolation, PixelShuffleRect
from .convnext import ConvNeXtBlock, CustomConv2d


def get_norm(norm, **kwargs):
    if norm == "none":
        return nn.Identity
    elif norm == "layernorm":
        return partial(nn.LayerNorm, eps=1e-6, **kwargs)
    elif norm == "layernorm-no-affine":
        return partial(nn.LayerNorm, elementwise_affine=False, eps=1e-6, **kwargs)
    else:
        raise NotImplementedError


class Conv2d_Q(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(Conv2d_Q, self).__init__(*kargs, **kwargs)


def conv2d_quantize_fn(bit):
    class Conv2d_Q_(Conv2d_Q):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                     bias=True):
            super(Conv2d_Q_, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                            bias)
            self.w_bit = bit
            self.quantize_fn = weight_quantize_fn(self.w_bit)

        def forward(self, input, order=None):
            weight_q = self.quantize_fn(self.weight)
            bias_q = self.quantize_fn(self.bias)
            return F.conv2d(input, weight_q, bias_q, self.stride, self.padding, self.dilation, self.groups)

    return Conv2d_Q_

class NervBlock(nn.Module):
    def __init__(self,dim_in,dim_out,num_groups,activation='gelu',**kwargs):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_groups = num_groups

        self.params = kwargs

        self.up_sample = self.params.get('up_sample', 2)
        
        self.activation = get_activation(activation, **kwargs)

        self.params = kwargs
        ks = self.params.get('ks', 3)
        strd = self.params.get('strd', 1)
        conv_type = self.params.get('conv_type', 'standard')
        wbit = self.params.get('wbit', 32)
        padding = self.params.get('padding', math.ceil(ks / 2))
        if wbit < 32:
            Conv2d = conv2d_quantize_fn(wbit)
        else:
            Conv2d = CustomConv2d

        if 'dec_block' not in self.params or self.params['dec_block']:
            if conv_type == 'standard' or conv_type == 'conv':
                if type(self.up_sample) == int:
                    self.up_sample = (self.up_sample, self.up_sample)
                conv = Conv2d(self.dim_in,self.dim_out,\
                                    kernel_size=(ks,ks),stride=(1,1),padding=(math.ceil((ks - 1) // 2),math.ceil((ks - 1) // 2)),\
                                    groups=self.num_groups)
                self.nerv_block = nn.Sequential(conv,\
                                    PixelShuffleRect(self.up_sample[0], self.up_sample[1]),\
                                    self.activation)
            elif conv_type == 'compact':
                if type(self.up_sample) == int:
                    self.up_sample = (self.up_sample, self.up_sample)
                conv1 = Conv2d(self.dim_in, self.dim_out, kernel_size=(ks,ks), stride=(1,1), padding=(math.ceil((ks - 1) // 2),math.ceil((ks - 1) // 2)),groups=self.num_groups)
                conv2 = Conv2d(self.dim_out, self.dim_out, kernel_size=1)
                self.nerv_block = nn.Sequential(conv1,\
                                    conv2,\
                                    PixelShuffleRect(self.up_sample[0], self.up_sample[1]),\
                                    self.activation)
        else:
            conv = Conv2d(self.dim_in, self.dim_out, ks, strd, padding)
            self.nerv_block = nn.Sequential(conv, self.activation)

    def forward(self, x):
        if len(x.shape) == 5:
            T = x.shape[1]
            x = einops.rearrange(x, 'n t h w c -> (n t) c h w')
            x = self.nerv_block(x)
            x = einops.rearrange(x, '(n t) c h w -> n t h w c', t=T)
            return x
        else:
            return self.nerv_block(x)


class ConvUpBlock(nn.Module):
    def __init__(self,dim_in,dim_out,stride,activation='gelu',**kwargs):
        super().__init__()
        if type(stride) == int:
            stride = (stride, stride)
        wbit = kwargs.get('wbit', 32)
        if wbit < 32:
            Conv2d = conv2d_quantize_fn(wbit)
        else:
            Conv2d = CustomConv2d

        if dim_in <= dim_out:
            factor = 4
            conv1 = Conv2d(dim_in, stride[0] * stride[1] * (dim_in // factor), 3, 1, 1, bias=True)
            conv2 = Conv2d(dim_in // factor, dim_out, 3, 1, 1, bias=True)
        else:
            conv1 = Conv2d(dim_in, dim_out, 3, 1, 1, bias=True)
            conv2 = Conv2d(dim_out, dim_out * stride[0] * stride[1], 3, 1, 1, bias=True)
        self.up_scale = PixelShuffleRect(stride[0], stride[1])
        self.activation = get_activation(activation, **kwargs)
        if dim_in <= dim_out:
            self.nerv_block = nn.Sequential(conv1, self.up_scale, conv2, self.activation)
        else:
            self.nerv_block = nn.Sequential(conv1, conv2, self.up_scale, self.activation)

    def forward(self, x):
        return self.nerv_block(x)


class HiNeRVUpsampler(nn.Module):
    """
    HiNeRV Upsampler. It combined the upsampling together with the encoding and cropping.
    """
    def __init__(self, channels, scale, upsample_type, upsample_method):
        super().__init__()
        self.channels = channels
        self.scale = scale
        self.upsample_type = upsample_type

        self.norm = get_norm('layernorm-no-affine')(self.channels) ### NOTE: hardcoded, HiNeRV technically had options but always used this from what I could tell

        # Layer
        if self.upsample_type == 'trilinear':
            self.layer = FastTrilinearInterpolation(upsample_method)
        else:
            raise ValueError

    def extra_repr(self):
        s = 'scale={scale}, upsample_type={upsample_type}'
        return s.format(**self.__dict__)

    def forward(self, x: torch.Tensor, idx: torch.IntTensor, idx_max: tuple[int, int, int],
                size: tuple[int, int, int], scale: tuple[int, int, int], padding: tuple[int, int, int], patch_mode: bool=True, mask: Optional[torch.Tensor]=None,
                skip_norm: bool=False):
        """
        During the forward pass, the input tensor will be upscale by the 'scale' factor, then a cropping will be applied to reduce the size to 'output_size'.

        Inputs:
            x: input tensor with shape [N, T1, H1, W1, C]
            idx: patch index tensor with shape [N, 3]
            idx_max: list of 3 ints. Represents the range of patch indexes.
            size: list of 3 ints. Represents the size of the fulle video. It does not have to be the same as the input size, as the input can be a patch from the full video.
            scale: list of 3 ints. Represents the scale factor. This will be used to compute the output size.
            padding: list of 3 ints. Represents the padding size. This will be used to compute the output size.
            patch_mode: if True, the input is a patch from the full video, and the faster implementation will be used.
            mask: mask tensor with shape [N, T1, H1, W1, C]. If not None, it will be multiplied to the output.

        Output:
            a tensor with shape [N, T2, H2, W2, C]
        """
        assert x.ndim == 5, x.shape
        assert idx.ndim == 2 and idx.shape[1] == 3, idx.shape
        assert len(idx_max) == 3
        assert len(scale) == 3
        assert len(size) == 3
        assert len(padding) == 3

        N, T_in, H_in, W_in, C = x.shape
        T_out, H_out, W_out = tuple(size[d] // idx_max[d] + 2 * padding[d] for d in range(3))
        assert (T_out - T_in * scale[0]) % 2 == (H_out - H_in * scale[1]) % 2 == (W_out - W_in * scale[2]) % 2 == 0, 'Under this configuration, padding is not symmetric and can cause problems!'

        if not skip_norm:
            x = self.norm(x)

        if self.upsample_type in ['trilinear', 'nearest']:
            x = self.layer(x, idx, idx_max, size, scale, padding, patch_mode)
            assert_shape(x, (N, T_out, H_out, W_out, C))
        else:
            raise NotImplementedError

        return x
    

class HiNeRVBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, ks):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.main_block = ConvNeXtBlock(dim_in=dim_in, dim_out=dim_out, dim_hidden=dim_hidden, 
                                              norm='layernorm-no-affine', kernel_size=ks, padding=ks//2, drop_path=0.0, 
                                              layer_scale_init_value=0, bias=False)
        
    def forward(self, input, mask=None):
        x = self.main_block(input)

        if mask is not None:
            x = mask * x

        if self.dim_in == self.dim_out:
            x = x + input

        return x
    
class ToStyle(nn.Module):
    def __init__(self, in_chan, out_chan, bias=True):
        super().__init__()
        self.conv = nn.Conv3d(in_chan, out_chan*2, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=bias)

    def forward(self, x, style):
        B, C, T, H, W = x.shape
        style = self.conv(style)  # style -> [B, 2*C, 1, H, W]
        style = style.view(B, 2, C, -1, H, W)  # [B, 2, C, 1, H, W]
        x = x * (style[:, 0] + 1.) + style[:, 1] # [B, C, T, H, W]
        return x
    
class PixelShuffleTri(nn.Module):
    def __init__(self, scale=(1, 2, 2)):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        B, C, T, H, W = x.size()
        C_out = C // (self.scale[0] * self.scale[1] * self.scale[2])
        x = x.view(B, C_out, self.scale[0], self.scale[1], self.scale[2], T, H, W)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(B, C_out, T * self.scale[0], H * self.scale[1], W * self.scale[2])
        return x

class DivNeRVPredictionHead(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = nn.Sequential(
                        nn.Conv3d(in_chan, in_chan // 4, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True),
                        nn.GELU(),
                        nn.Conv3d(in_chan // 4, out_chan, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True),
                    )

    def forward(self, x):
        x = self.conv(x)
        return x

class DivNeRVBlock(nn.Module):
    def __init__(self, kernel=3, bias=True, **kwargs):
        super().__init__()
        in_chan = kwargs['ngf']
        out_chan = kwargs['new_ngf'] * kwargs['stride'] * kwargs['stride']
        # Spatially-adaptive Fusion
        self.to_style = ToStyle(64, in_chan, bias=bias)
        # 3x3 Convolution-> PixelShuffle -> Activation, same as NeRVBlock
        self.conv = nn.Conv3d(in_chan, out_chan, kernel_size=(1, kernel, kernel), stride=(1, 1, 1), padding=(0, kernel//2, kernel//2), bias=bias)
        self.upsample = PixelShuffleTri(scale=(1, kwargs['stride'], kwargs['stride']))
        self.act = nn.GELU()
        # Global Temporal MLP module
        self.tfc = nn.Conv2d(kwargs['new_ngf']*kwargs['clip_size'], kwargs['new_ngf']*kwargs['clip_size'], 1, 1, 0, bias=True, groups=kwargs['new_ngf'])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Conv3d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)

    def forward(self, x, style_appearance):
        x = self.to_style(x, style_appearance)
        x = self.act(self.upsample(self.conv(x)))
        B, C, D, H, W = x.shape
        x = x + self.tfc(x.view(B, C*D, H, W)).view(B, C, D, H, W)
        return x