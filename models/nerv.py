### code loosely adapted from https://github.com/abhyantrika/mediainr, and ###
### https://github.com/haochen-rye/HNeRV ###

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.layers import NervBlock, NeRVMLP, TransformerBlock, ConvUpBlock, conv2d_quantize_fn, HiNeRVUpsampler, HiNeRVBlock, DivNeRVBlock, DivNeRVPredictionHead
from models.layers.positional_encoders import PosEncoding, PosEncodingHiNeRVLocal, DivNeRVFrameIndexPositionalEncoding
from models.layers.warpkeyframe import WarpKeyframe
from utils import assert_shape, compute_pixel_idx_3d, compute_paddings, crop_tensor_nthwc, get_encoding_cfg, PixelShuffleRect
from typing import Sequence
from typing import Optional
import numpy as np
import einops


def OutImg(x, out_bias:str='tanh'):
    if out_bias == 'sigmoid':
        return torch.sigmoid(x)
    elif out_bias == 'tanh':
        return (torch.tanh(x) * 0.5) + 0.5
    else:
        return x + float(out_bias)


class NeRV(torch.nn.Module):
    def __init__(self, ch_in:int, fc_h:int, fc_w:int, fc_dim:int, block_dim:int, dec_strds:Sequence[int], ks_1:int, ks_2:int, reduce:int, lower_width:int, activation, out_bias:bool, expansion:Optional[int]=None, stem_type:str='hnerv', net_config:dict={}, **kwargs):
        super().__init__()

        self.fc_dim = fc_dim
        self.fc_h, self.fc_w = fc_h, fc_w
        self.stem_type = stem_type
        self.use_fuse_t = net_config.get('use_fuse_t', False)
        self.use_norm = net_config.get('use_norm', False)
        self.use_old_upconv = net_config.get('use_old_upconv', False)
        self.use_compact_blocks = net_config.get('use_compact_blocks', False)
        self.wbit = net_config.get('wbit', 32)
        self.paddings = net_config.get('paddings', None)
        self.stem_paddings = net_config.get('stem_paddings', None)
        self.use_linear_head = net_config.get('use_linear_head', False)
        self.use_hinerv_upsamplers = net_config.get('use_hinerv_upsamplers', False)
        self.decoding_blocks = net_config.get('decoding_blocks', None)
        self.zero_bias_init = net_config.get('zero_bias_init', False)
        self.base_size = net_config.get('base_size', None)
        self.input_channels = ch_in
        self.local_encoding_config = net_config.get('local_encoding_config', None)
        self.positional_encoding = net_config.get('positional_encoding', None)
        self.use_ccu = net_config.get('use_ccu', False)
        self.net_config = net_config
        self.flow_block_offset = 3
        padding_list = net_config.get('padding_list', None)
        if self.wbit < 32:
            Conv2d = conv2d_quantize_fn(self.wbit)
        else:
            Conv2d = torch.nn.Conv2d

        
        self.no_prune_prefix = ('encoding',) + tuple(k for k, v in self.named_modules() if isinstance(v, PosEncoding))
        self.no_quant_prefix = ()
        if self.positional_encoding in ['enerv', 'hnerv', 'henerv']:
            self.no_quant_prefix = ('positional_encoder')

        # The bitstream is the part of data that will be transmitted/stored.
        bitstream_prefix = []

        self.scales = [[1, strd, strd] if type(strd) == int else [1, strd[0], strd[1]] for strd in dec_strds]
        self.min_patch_size = tuple(np.prod(np.array(self.scales), axis=0).tolist()) if self.scales else None
        if self.paddings is not None:
            if tuple(self.paddings) == (-1, -1, -1):
                assert tuple(self.stem_paddings) == (-1, -1, -1), 'both padding must be set/not set at the same time'
                paddings = compute_paddings(output_patchsize=(1, math.prod(dec_strds), math.prod(dec_strds)),
                                            scales=self.scales, kernel_sizes=tuple((0, min(ks_1+2*i, ks_2), min(ks_1+2*i, ks_2)) for i in range(len(self.scales))),
                                            depths=self.decoding_blocks)
                self.stem_paddings = tuple(paddings[0][d] + (0 if d == 0 else (net_config['stem_ks'] - 1) // 2) for d in range(3))
                self.paddings = paddings[1:]
            else:
                assert tuple(self.stem_paddings) != (-1, -1, -1), 'both padding must be set/not set at the same time'
                assert all(p >= 0 for p in self.stem_paddings) and all(p >= 0 for p in self.paddings)
                self.stem_paddings = self.stem_paddings
                self.paddings = [self.paddings for _ in range(len(self.decoding_blocks))]

        # BUILD Decoder LAYERS        
        ngf = fc_dim
        if stem_type == 'hnerv':
            out_f = int(ngf * self.fc_h * self.fc_w)
            self.stem = NervBlock(dim_in=ch_in, dim_out=out_f, num_groups=1, activation=activation, dec_block=False, ks=1, strd=1, padding=0)
            bitstream_prefix.append('stem')
        elif stem_type == 'diff_nerv':
            c2_dim = net_config['dec_dim']
            #padding_list = net_config['padding_list']
            conv_type = net_config['conv_type']
            diff_dec_stride = net_config['diff_dec_stride']
            diff_dec_padding = net_config['diff_dec_padding']
            diff_dec_kernel = net_config['diff_dec_kernel']

            self.diff_enc_layers = nn.ModuleList()
            self.diff_dec_layers = nn.ModuleList()
            bitstream_prefix.append('diff_enc_layers')
            bitstream_prefix.append('diff_dec_layers')

            ngf = int(int(c2_dim/1.2)/1.2)
            self.diff_exc_layer = nn.Conv2d(2, ngf, kernel_size=1, stride=1)
            bitstream_prefix.append('diff_exc_layer')
            for i, stride in enumerate(diff_dec_stride):
                new_ngf = round(ngf//1.2)
                self.diff_dec_layers.append(
                    NervBlock(dim_in=ngf, dim_out=new_ngf * stride[0] * stride[1], num_groups=1, up_sample=stride, 
                            activation=activation, ks=diff_dec_kernel[i], padding=diff_dec_padding[i], conv_type=conv_type))
                ngf = new_ngf

            ngf = c2_dim
            ngf_a = int(int(int(c2_dim/reduce)/reduce)/reduce)
            ngf_a = int(int(int(c2_dim/reduce)/reduce)/reduce)
            self.dec_p_c = nn.Conv2d(ngf_a, ngf_a, kernel_size=3, stride=1, padding=1)
            self.dec_p_d = nn.Conv2d(ngf_a, ngf_a, kernel_size=3, stride=1, padding=1)

            self.dec_s_c = nn.Conv2d(ngf_a, ngf_a, kernel_size=3, stride=1, padding=1)
            self.dec_s_d = nn.Conv2d(ngf_a, ngf_a, kernel_size=3, stride=1, padding=1)
            bitstream_prefix.append('dec_p_c')
            bitstream_prefix.append('dec_p_d')
            bitstream_prefix.append('dec_s_c')
            bitstream_prefix.append('dec_s_d')
        elif stem_type == "div_nerv":
            stride_list = net_config.get("stride_list")
            embed = net_config.get('embed')
            clip_size = net_config.get("clip_size", 8)
            device = net_config.get("device")
            expansion = net_config.get("expansion")
            lower_width = net_config.get("lower_width") 

            ngf = fc_dim
            self.PE = DivNeRVFrameIndexPositionalEncoding(embed)
            self.stem = NeRVMLP(dim_list=[self.PE.embed_length, 512, fc_h * fc_w * ngf], act="gelu", bias=True)
            bitstream_prefix.append('PE')
            bitstream_prefix.append('stem')

            self.stride_list = stride_list
            self.num_stages = len(self.stride_list)

            encoder_dim = 64
            self.norm = nn.InstanceNorm3d(ngf + encoder_dim)
            bitstream_prefix.append('norm')

            self.decoder_list = nn.ModuleList()
            self.flow_pred_list = nn.ModuleList([DivNeRVPredictionHead(ngf + encoder_dim, 4)])
            bitstream_prefix.append('decoder_list')
            bitstream_prefix.append('flow_pred_list')

            height = fc_h
            width = fc_w
            self.wk_list = nn.ModuleList([WarpKeyframe(height, width, clip_size, device=device)])
            bitstream_prefix.append('wk_list')
            for i, stride in enumerate(self.stride_list):
                if i == 0:
                    new_ngf = int(ngf * expansion)
                else:
                    new_ngf = max(round(ngf / stride), lower_width)

                self.decoder_list.append(DivNeRVBlock(ngf=ngf + encoder_dim if i == 0 else ngf, new_ngf=new_ngf, 
                                                    stride=stride, clip_size=clip_size))
                self.flow_pred_list.append(DivNeRVPredictionHead(new_ngf, 4))
                height = height * stride
                width = width * stride
                self.wk_list.append(WarpKeyframe(height, width, clip_size, device=device))

                ngf = new_ngf
            
            self.rgb_head_layer = DivNeRVPredictionHead(new_ngf + 3, 3)
            bitstream_prefix.append('rgb_head_layer')

            self.dataset_mean = torch.tensor(net_config['dataset_mean']).view(1, 3, 1, 1, 1).cuda()
            self.dataset_std = torch.tensor(net_config['dataset_std']).view(1, 3, 1, 1, 1).cuda()
        elif stem_type == 'enerv':
            stem_dim_list = [int(x) for x in net_config['stem_dim_num'].split('_')]
            t_ch_in = net_config.get('t_ch_in', ch_in)
            self.stem_t = NeRVMLP(dim_list=[t_ch_in] + stem_dim_list + [block_dim], act=activation)
            self.stem_xy = NeRVMLP(dim_list=[ch_in*2, block_dim], act=activation)
            self.trans1 = TransformerBlock(
                dim=block_dim, heads=1, dim_head=64, mlp_dim=net_config['enerv_mlp_dim'], dropout=0., prenorm=False
            )
            self.trans2 = TransformerBlock(
                dim=block_dim, heads=8, dim_head=64, mlp_dim=net_config['enerv_mlp_dim'], dropout=0., prenorm=False
            )
            bitstream_prefix.append('stem_t')
            bitstream_prefix.append('stem_xy')
            bitstream_prefix.append('trans1')
            bitstream_prefix.append('trans2')
            if block_dim == fc_dim:
                self.toconv = torch.nn.Identity()
            else:
                self.toconv = NeRVMLP(dim_list=[block_dim, fc_dim], act=activation)
                bitstream_prefix.append('toconv')
        elif stem_type == 'nerv':
            stem_dim, stem_num = [int(x) for x in net_config['stem_dim_num'].split('_')]
            mlp_dim_list = [ch_in] + [stem_dim] * stem_num + [fc_h * fc_w * fc_dim]
            self.stem = NeRVMLP(dim_list=mlp_dim_list, act=activation)
            bitstream_prefix.append('stem')
        elif stem_type == 'hinerv':
            self.stem = NervBlock(dim_in=ch_in, dim_out=ngf, num_groups=1, activation='none', dec_block=False,
                                 ks=net_config['stem_ks'], strd=1, padding=net_config['stem_ks']//2, bias=True)
            bitstream_prefix.append('stem')

        if self.use_fuse_t:
            self.t_branch = NeRVMLP(dim_list=[kwargs['pe_t_manipulate_embed_length'], 128, 128], act=activation)
            bitstream_prefix.append('t_branch')

        upsample_layers = []
        positional_encoder_layers = []
        norm_layers = []
        t_layers = []
        decoder_blocks = []
        self.flow_layer = None
        for i, strd in enumerate(dec_strds):                         
            reduction = math.sqrt(strd) if reduce==-1 else reduce
            if i == 0 and expansion is not None and not self.use_hinerv_upsamplers:
                new_ngf = int(ngf * expansion[i])
                if self.local_encoding_config is not None:
                    T1, H1, W1 = (self.base_size[d] for d in range(3))
                    T2, H2, W2 = T1, H1 * strd, W1 * strd
            elif i == 0 and self.use_hinerv_upsamplers:
                new_ngf = int(max(ngf, lower_width))
                if self.local_encoding_config is not None:
                    T1, H1, W1 = (self.base_size[d] for d in range(3))
                    T2, H2, W2 = T1, H1 * strd, W1 * strd
            elif i == 0:
                if stem_type == 'diff_nerv':
                   new_ngf = round(ngf // reduction)
                else: 
                    new_ngf = int(max(round(ngf / reduction), lower_width))
                if self.local_encoding_config is not None:
                    T1, H1, W1 = (self.base_size[d] for d in range(3))
                    T2, H2, W2 = T1, H1 * strd, W1 * strd
            else:
                if stem_type == 'diff_nerv':
                   new_ngf = round(ngf // reduction)
                else: 
                    new_ngf = int(max(round(ngf / reduction), lower_width))
                if self.local_encoding_config is not None:
                    T1, H1, W1 = T2, H2, W2
                    T2, H2, W2 = T1, H1 * strd, W1 * strd

            if self.use_compact_blocks:
                if i < len(dec_strds) - 1:
                    if new_ngf % 8 > 0:
                        new_ngf = new_ngf + (8 - (new_ngf % 8))
                else:
                    if (new_ngf * (strd*strd if type(strd) == int else strd[0]*strd[1])) % 8 > 0:
                        new_ngf = new_ngf + (8 - (new_ngf % 8))   
            if self.use_norm:
                norm_layers.append(torch.nn.InstanceNorm2d(ngf, affine=False))
            if self.use_fuse_t:
                t_layers.append(NeRVMLP(dim_list=[128, 2*ngf], act=activation))
            if self.use_hinerv_upsamplers:
                upsample_layers.append(HiNeRVUpsampler(channels=ngf, scale=strd, upsample_type='trilinear', upsample_method='matmul-th-w'))
                
                if self.local_encoding_config is not None:
                    _encoding_config = get_encoding_cfg('upsample', i, size=(T2, H2, W2, ngf), **self.local_encoding_config)
                    positional_encoder_layers.append(PosEncodingHiNeRVLocal(scale=self.scales[i], channels=ngf, cfg=_encoding_config))

                cur_layers = []
                for j in range(self.decoding_blocks[i]):
                    cur_layer = HiNeRVBlock(dim_in=ngf if j == 0 else new_ngf, dim_out=new_ngf, dim_hidden=int(new_ngf * expansion[i]), ks=min(ks_1+2*i, ks_2))
                    cur_layers.append(cur_layer)
                decoder_blocks.append(torch.nn.ModuleList(cur_layers))
            else:
                if i == 0 and self.use_old_upconv:
                    cur_blk = ConvUpBlock(dim_in=ngf, dim_out=new_ngf, stride=strd, activation=activation, wbit=self.wbit)
                else:
                    if i > 0 and self.use_compact_blocks:
                        cur_blk = NervBlock(dim_in=ngf, dim_out=new_ngf*(strd*strd if type(strd) == int else strd[0] * strd[1]), num_groups=8, activation=activation, dec_block=True, 
                                            ks=min(ks_1+2*i, ks_2), up_sample=strd, conv_type='compact', wbit=self.wbit)
                    # elif self.is_diff_nerv:
                    #     cur_blk = NervBlock(dim_in=ngf, dim_out=new_ngf*(strd*strd if type(strd) == int else strd[0] * strd[1]), num_groups=1, activation=activation, 
                    #                         ks=kernel_list[i], up_sample=strd, padding=padding_list[i], conv_type=conv_type)
                    else:
                        cur_blk = NervBlock(dim_in=ngf, dim_out=new_ngf*(strd*strd if type(strd) == int else strd[0] * strd[1]), num_groups=1, activation=activation, dec_block=True, 
                                            ks=min(ks_1+2*i, ks_2), up_sample=strd, padding=padding_list[i] if padding_list else None, wbit=self.wbit)
                upsample_layers.append(cur_blk)

                if self.local_encoding_config is not None:
                    _encoding_config = get_encoding_cfg('upsample', i, size=(T2, H2, W2, ngf), **self.local_encoding_config)
                    positional_encoder_layers.append(PosEncodingHiNeRVLocal(scale=(1, 1, 1), channels=ngf, cfg=_encoding_config))

            ngf = new_ngf

            if 'flow_agg_idxs' in net_config and net_config['is_ffnerv']:
                if 'flow_block_offset' in net_config:
                    self.flow_block_offset = net_config['flow_block_offset']
                else:
                    self.flow_block_offset = 3
                if i == len(dec_strds) - self.flow_block_offset:
                    self.flow_layer = Conv2d(ngf, len(net_config['flow_agg_idxs'])*3, 1, 1, 0)
                    bitstream_prefix.append('flow_layer')
        
        self.upsample_layers = torch.nn.ModuleList(upsample_layers)
        self.positional_encoder_layers = torch.nn.ModuleList(positional_encoder_layers)
        self.norm_layers = torch.nn.ModuleList(norm_layers)
        self.t_layers = torch.nn.ModuleList(t_layers)
        self.decoder_blocks = torch.nn.ModuleList(decoder_blocks)
        bitstream_prefix.append('upsample_layers')
        bitstream_prefix.append('positional_encoder_layers')
        bitstream_prefix.append('norm_layers')
        bitstream_prefix.append('t_layers')
        bitstream_prefix.append('decoder_blocks')
        if self.use_linear_head:
            if 'flow_agg_idxs' in net_config and net_config['is_ffnerv']:
                dim_out = 5
            else:
                dim_out = 3
            self.head_layer = NervBlock(dim_in=ngf, dim_out=dim_out, num_groups=1, activation='sigmoid', dec_block=False, ks=1, strd=1, padding=0) 
            bitstream_prefix.append('head_layer')
        elif 'flow_agg_idxs' in net_config and net_config['is_ffnerv']:
            self.head_layer = Conv2d(ngf, 5, 1, 1, 0)
            bitstream_prefix.append('head_layer')
        elif self.stem_type == "div_nerv":
            self.head_layer = nn.Identity()
        else:
            self.head_layer = Conv2d(ngf, 3, 3, 1, 1) 
            bitstream_prefix.append('head_layer')
        self.out_bias = out_bias

        if self.zero_bias_init:
            if self.stem_type == 'enerv':
                self.stem_t.apply(self._init_blocks)
                self.stem_xy.apply(self._init_blocks)
                self.trans1.apply(self._init_blocks)
                self.trans2.apply(self._init_blocks)
                self.toconv.apply(self._init_blocks)
            elif self.stem_type == 'ffnerv':
                pass
            else:
                self.stem.apply(self._init_blocks)
            self.upsample_layers.apply(self._init_blocks)
            self.norm_layers.apply(self._init_blocks)
            self.t_layers.apply(self._init_blocks)
            self.decoder_blocks.apply(self._init_blocks)
            self.head_layer.apply(self._init_blocks)
            if self.flow_layer is not None:
                self.flow_layer.apply(self._init_blocks)

        if self.positional_encoding not in ['enerv', 'hnerv', 'henerv']:
            bitstream_prefix.append('positional_encoder')

        self.bitstream_prefix = tuple(bitstream_prefix)

    
    def _init_blocks(self, m):
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    

    def fuse_t(self, x, t):
        # x: [B, C, H, W], normalized among C
        # t: [B, 2* C]
        f_dim = t.shape[-1] // 2
        gamma = t[:, :f_dim]
        beta = t[:, f_dim:]

        gamma = gamma[..., None, None]
        beta = beta[..., None, None]
        out = x * gamma + beta
        return out
    

    def get_input_padding(self, patch_mode: bool=True):
        return self.stem_paddings if patch_mode else (self.stem_paddings[0], 0, 0)


    def forward(self, input):
        embed_quant_level = input.get('embed_quant_level', None)
        ### POSITIONAL ENCODING
        patch_mode = False
        if self.use_hinerv_upsamplers or self.stem_type == 'hinerv':
            idx_max = input['hinerv_sizes']['idx_max']
            patch_mode = input['patch_mode']
            v_size_in = v_size_out = self.base_size

            if patch_mode:
                assert all((input['hinerv_sizes']['patch_size'][d] % self.min_patch_size[d] == 0) for d in range(3))
                input_padding = self.get_input_padding(patch_mode=patch_mode)
                p_size_in = p_size_out = tuple(v_size_out[d] // idx_max[d] for d in range(3))
                p_size_out_padded = tuple(p_size_out[d] + 2 * input_padding[d] for d in range(3))
                _, px_mask = compute_pixel_idx_3d(input['t'], idx_max, v_size_out, input_padding, clipped=False, return_mask=True)
                px_mask_3d = px_mask[0][:, :, None, None, None] \
                                * px_mask[1][:, None, :, None, None] \
                                * px_mask[2][:, None, None, :, None]
            else:
                px_mask_3d = None
        elif self.local_encoding_config is not None:
            idx_max = input['hinerv_sizes']['idx_max'] ## NOTE: see if there is an issue with hardcoding 600
            v_size_in = v_size_out = self.base_size
        
        if self.positional_encoding in ['hinerv', 'hienerv']:
            input_padding = self.get_input_padding(patch_mode=patch_mode)
            if patch_mode:
                times = input['t']
            else:
                times = input['frame_id']
                zeros = torch.zeros_like(times)
                times = torch.cat((times, zeros, zeros), dim=-1)
            embed_dict = self.positional_encoder(times, idx_max, padding=input_padding)
        elif self.positional_encoding == "diff_nerv":
            embed_dict = self.positional_encoder(input['image'])
        elif self.positional_encoding == 'div_nerv':
            embed_dict = self.positional_encoder(input['keyframe'].squeeze(dim=1))
        elif self.positional_encoding == 'henerv':
            embed_dict = self.positional_encoder(input['image'].squeeze(dim=1), input['t'].squeeze(dim=1), qbit=embed_quant_level)
        elif self.positional_encoding == 'hnerv':
            embed_dict = self.positional_encoder(input['image'].squeeze(dim=1), qbit=embed_quant_level)
        elif self.positional_encoding == 'enerv':
            if input['t'].shape[1] == 3:
                embed_dict = self.positional_encoder(input['t'], input['hinerv_sizes']['idx_max'])
            else:
                times = input['t'].squeeze(dim=1)
                embed_dict = self.positional_encoder(times)
        else:
            if input['t'].shape[1] == 3:
                #print(input['hinerv_sizes']['idx_max'])
                #print(input['t'])
                times = (input['t'][:,0]*input['hinerv_sizes']['idx_max'][1]*input['hinerv_sizes']['idx_max'][2] + \
                        input['t'][:,1]*input['hinerv_sizes']['idx_max'][2] + \
                        input['t'][:,2]) / \
                        (input['hinerv_sizes']['idx_max'][0] * input['hinerv_sizes']['idx_max'][1] * input['hinerv_sizes']['idx_max'][2])
                #print(times)
            else:
                times = input['t'].squeeze(dim=1)
            embed_dict = self.positional_encoder(times)
        
        ### STEM
        if self.stem_type == 'enerv':
            t_embed, xy_embed = embed_dict['t_embed'], embed_dict['xy_embed']
            t_embed = self.stem_t(t_embed.squeeze(2, 3))
            if xy_embed.shape[0] != t_embed.shape[0]:
                xy_embed = self.stem_xy(xy_embed).unsqueeze(0).expand(t_embed.shape[0], -1, -1)
            else:
                xy_embed = self.stem_xy(xy_embed)
            xy_embed = self.trans1(xy_embed)
            t_embed_list = [t_embed for _ in range(xy_embed.shape[1])]
            t_embed_map = torch.stack(t_embed_list, dim=1)  # [B, h*w, L]
            embed = xy_embed * t_embed_map
            embed = self.toconv(self.trans2(embed))
            embed = embed.reshape(embed.shape[0], self.fc_h, self.fc_w, embed.shape[-1])
            embed = embed.permute(0, 3, 1, 2)
            output = embed
            if self.use_hinerv_upsamplers:
                output = einops.rearrange(output, 'b c h w -> b 1 h w c')
        elif self.stem_type == 'diff_nerv':
            diff = embed_dict['diff']
            output = embed_dict['output']
        elif self.stem_type == "div_nerv":
            embed_input = self.PE(input['t'])

            B, C, D = embed_input.size()
            backward_distance = input['backward_distance'].view(B, 1, -1, 1, 1)
            forward_distance = 1 - backward_distance

            key_feature_list = embed_dict

            embed_input = embed_input.permute(0, 2, 1)
            embed_input = embed_input.reshape(B, D, -1) 
            output = self.stem(embed_input)  # [B, C*fc_h*fc_w, D]

            output = output.view(B, -1, self.fc_h, self.fc_w, D).permute(0, 1, 4, 2, 3)  # [B, C, D, fc_h, fc_w]
            content_feature = F.interpolate(key_feature_list[0], scale_factor=(D/2, 1, 1), mode='trilinear') # [B, encoder_dim, D, fc_h, fc_w]
            output = self.norm(torch.concat([output, content_feature], dim=1))

            for i in range(self.num_stages + 1):
                # generate flow at the decoder input stage
                flow = self.flow_pred_list[i](output) # [B, 4, D, fc_h, fc_w]
                forward_flow, backward_flow = torch.split(flow, [2, 2], dim=1)
                start_key_feature, end_key_feature = torch.split(key_feature_list[i], [1, 1], dim=2)
                # warp the keyframe features with predicted forward and backward flow
                forward_warp = self.wk_list[i](start_key_feature, forward_flow)
                backward_warp = self.wk_list[i](end_key_feature, backward_flow)
                # distance-aware weighted sum
                fused_warp = forward_warp * forward_distance + backward_warp * backward_distance # (1 - t) * forward_warp + t * backward_warp

                if i < self.num_stages:
                    output = self.decoder_list[i](output, fused_warp)
                else:
                    output = self.rgb_head_layer(torch.cat([output, fused_warp], dim=1))

            output = output * self.dataset_std + self.dataset_mean
            output = output.clamp(min=0, max=1)
        else:
            t_embed = embed_dict['t_embed']
            if self.stem_type == 'nerv':
                output = self.stem(t_embed.squeeze(2, 3)).view(t_embed.shape[0], self.fc_dim, self.fc_h, self.fc_w)
            elif self.stem_type == 'ffnerv':
                output = t_embed
                if self.use_hinerv_upsamplers:
                    output = einops.rearrange(output, 'b c h w -> b 1 h w c')
            elif self.stem_type == 'hinerv':
                if patch_mode:
                    assert_shape(t_embed, (input['t'].shape[0],) + p_size_out_padded + (self.input_channels,))

                output = self.stem(t_embed)
                if px_mask_3d is not None:
                    output = output * px_mask_3d

                if not self.use_hinerv_upsamplers:
                    output = output.squeeze(1).permute(0,3,1,2)
            else:
                output = self.stem(t_embed)
                if not self.use_hinerv_upsamplers:
                    n, _, h, w = output.shape
                    output = output.view(n, -1, self.fc_h, self.fc_w, h, w).permute(0,1,4,2,5,3).reshape(n,-1,self.fc_h * h, self.fc_w * w)
                elif px_mask_3d is not None:
                    n, _, h, w = output.shape
                    output = output.view(n, -1, self.fc_h, self.fc_w, h, w).permute(0,1,4,2,5,3).reshape(n,p_size_out_padded[0],-1,self.fc_h * h, self.fc_w * w)
                    output = output.permute(0,1,3,4,2)
                    output = output * px_mask_3d
                else:
                    n, _, h, w = output.shape
                    output = output.view(n, -1, self.fc_h, self.fc_w, h, w).permute(0,1,4,2,5,3).reshape(n,1,-1,self.fc_h * h, self.fc_w * w)
                    output = output.permute(0,1,3,4,2)

        if self.use_fuse_t:
            t_manipulate_embed = embed_dict['t_manipulate_embed']
            t_manipulate = self.t_branch(t_manipulate_embed)
        
        for i, upsample_layer in enumerate(self.upsample_layers):
            if self.use_norm:
                if self.use_hinerv_upsamplers:
                    output = output.squeeze(1).permute(0,3,1,2)
                output = self.norm_layers[i](output)
                if self.use_hinerv_upsamplers:
                    output = output.permute(0,2,3,1).unsqueeze(1)

            if self.use_fuse_t:
                t_feat = self.t_layers[i](t_manipulate)
                if self.use_hinerv_upsamplers:
                    output = output.squeeze(1).permute(0,3,1,2)
                output = self.fuse_t(output, t_feat)
                if self.use_hinerv_upsamplers:
                    output = output.permute(0,2,3,1).unsqueeze(1)

            if (self.stem_type == "diff_nerv" or self.use_ccu) and i == 3:
                # ccu to fuse content and diff embedding 
                diff = self.diff_exc_layer(diff) 
                diff = self.diff_dec_layers[0](diff) 
                p = torch.tanh(self.dec_p_c(output) + self.dec_p_d(diff))
                s = torch.sigmoid(self.dec_s_c(output) + self.dec_s_d(diff))
                output = s * p + (1-s)*output

            if self.use_hinerv_upsamplers:
                padding = self.paddings[i] if patch_mode else (self.paddings[0][0], 0, 0)
                scale = self.scales[i]
                v_size_in = v_size_out
                v_size_out = tuple(int(v_size_in[d] * scale[d]) for d in range(3))
                if patch_mode:
                    p_size_in = p_size_out
                    p_size_in_padded = p_size_out_padded
                    p_size_out = tuple(int(p_size_in[d] * scale[d]) for d in range(3))
                    p_size_out_padded = tuple(p_size_out[d] + 2 * padding[d] for d in range(3))
                    
                    assert all(p_size_in_padded[d] * scale[d] >= p_size_out_padded[d] for d in range(3)), 'the input padding is too small'

                    _, px_mask = compute_pixel_idx_3d(input['t'], idx_max, v_size_out, padding, clipped=False, return_mask=True)
                    px_mask_3d = px_mask[0][:, :, None, None, None] \
                                    * px_mask[1][:, None, :, None, None] \
                                    * px_mask[2][:, None, None, :, None]
                else:
                    px_mask_3d = None

                times = input['t']
                if not patch_mode:
                    zeros = torch.zeros_like(times)
                    times = torch.cat((times, zeros, zeros), dim=-1)

                output = upsample_layer(output, idx=times, idx_max=idx_max, size=v_size_out, scale=scale, 
                               padding=padding, patch_mode=patch_mode, mask=px_mask_3d) ## NOTE: mask is not actually used in trilinear mode
                if self.local_encoding_config is not None:
                    output = self.positional_encoder_layers[i](output, idx=times, idx_max=idx_max, size=v_size_out, 
                                                               scale=scale, padding=padding)
                
                for decoder_block in self.decoder_blocks[i]:
                    output = decoder_block(output, px_mask_3d)
            else:
                if self.local_encoding_config is not None:
                    padding = self.paddings[i] if patch_mode else (self.paddings[0][0], 0, 0)
                    scale = self.scales[i]
                    v_size_in = v_size_out
                    v_size_out = tuple(int(v_size_in[d] * scale[d]) for d in range(3))
                    pre_padding = (0, int(padding[1] / 2), int(padding[2] / 2))

                    times = input['t']
                    if not patch_mode:
                        zeros = torch.zeros_like(times)
                        times = torch.cat((times, zeros, zeros), dim=-1)
                    output = output.permute(0,2,3,1).unsqueeze(1)
                    output = self.positional_encoder_layers[i](output, idx=times, idx_max=idx_max, size=v_size_in, 
                                                               scale=[1, 1, 1], padding=pre_padding)
                    output = output.squeeze(1).permute(0,3,1,2)
                output = upsample_layer(output)
                if patch_mode:
                    _, px_mask = compute_pixel_idx_3d(input['t'], idx_max, v_size_out, padding, clipped=False, return_mask=True)
                    px_mask_3d = px_mask[0][:, :, None, None, None] \
                                    * px_mask[1][:, None, :, None, None] \
                                    * px_mask[2][:, None, None, :, None]
                    output = px_mask_3d * output

            if i == len(self.upsample_layers) - self.flow_block_offset and self.flow_layer is not None:
                if self.use_hinerv_upsamplers:
                    output = output.squeeze(1).permute(0,3,1,2)
                flow_out = self.flow_layer(output)
                if self.use_hinerv_upsamplers:
                    output = output.permute(0,2,3,1).unsqueeze(1)
            
            if i > len(self.upsample_layers) - self.flow_block_offset and self.flow_layer is not None and patch_mode:
                if self.use_hinerv_upsamplers:
                    flow_out = flow_out.permute(0,2,3,1).unsqueeze(1)
                    flow_out = upsample_layer(flow_out, idx=times, idx_max=idx_max, size=v_size_out, scale=scale, 
                               padding=padding, patch_mode=patch_mode, mask=px_mask_3d, skip_norm=True) 
                    flow_out = flow_out.squeeze(1).permute(0,3,1,2)

        img_out = self.head_layer(output)

        if self.use_hinerv_upsamplers:
            if patch_mode:
                img_out = crop_tensor_nthwc(img_out, p_size_out)
                if self.flow_layer is not None:
                    flow_out = flow_out.permute(0,2,3,1).unsqueeze(1)
                    flow_out = crop_tensor_nthwc(flow_out, p_size_out)
                    flow_out = flow_out.permute(0,4,1,2,3).contiguous(memory_format=torch.channels_last_3d).squeeze(2)
                if 'flow_agg_idxs' in self.net_config and self.net_config['is_ffnerv']:
                    assert_shape(img_out, (input['t'].shape[0],) + p_size_out + (5,))
                else:
                    assert_shape(img_out, (input['t'].shape[0],) + p_size_out + (3,))

            img_out = img_out.permute(0, 4, 1, 2, 3).contiguous(memory_format=torch.channels_last_3d).squeeze(2)

        if 'flow_agg_idxs' in self.net_config and self.net_config['is_ffnerv']:
            img_out[:,0:3,:,:] = OutImg(img_out[:,0:3,:,:], self.out_bias)
        elif self.stem_type == "div_nerv":
            pass
        elif self.use_linear_head:
            pass
        else:
            img_out = OutImg(img_out, self.out_bias)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if self.flow_layer is not None:
            return img_out, flow_out
        else:
            return img_out
