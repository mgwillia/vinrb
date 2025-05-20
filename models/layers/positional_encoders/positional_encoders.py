### code adapted from https://github.com/abhyantrika/mediainr ###

import numpy as np
import math 

import torch 
import torch.nn as nn
import torch.nn.functional as F
import einops

from utils import FeatureGrid, SparseGrid, weight_quantize_fn, NormalizedCoordinate, \
	compute_pixel_idx_1d, compute_pixel_idx_3d, interpolate3D, GridTrilinear3D, \
	compute_best_quant_axis, _quantize_ste
from models.layers import ConvNeXt, ConvNeXtBlock, LayerNorm

from typing import Optional


class PosEncoding(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self):
		pass


class PosEncodingFourier(PosEncoding):
	def __init__(self,dim_in,mapping_size,scale):
		super().__init__()
		self.dim_in = dim_in
		self.mapping_size = mapping_size
		self.scale = scale

		self.register_buffer('B',torch.randn((self.mapping_size,self.dim_in)) * self.scale)
		self.output_dim = self.mapping_size*2

	def forward(self,x):
		x_proj = (2. * np.pi * x) @ self.B.t()
		return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
		

class PosEncodingNeRV(PosEncoding):
	def __init__(self, pe_lbase, pe_levels, manip_lbase=None, manip_levels=None):
		super().__init__()
		self.pe_bases = pe_lbase ** torch.arange(int(pe_levels)) * math.pi
		if manip_lbase is not None and manip_levels is not None:
			self.manip_bases = manip_lbase ** torch.arange(int(manip_levels)) * math.pi
		else:
			self.manip_bases = None

	def pos_embed_helper(self, pos, bases):
		pos = pos[:,None]
		value_list = pos * bases.to(pos.device)
		return torch.cat([torch.sin(value_list), torch.cos(value_list)], dim=-1)

	def forward(self, t):
		assert t.dim() == 1
		
		embed_dict = {'t_embed': self.pos_embed_helper(t, self.pe_bases).view(t.size(0), -1, 1, 1).float()}
		if self.manip_bases is not None:
			embed_dict['t_manipulate_embed'] = self.pos_embed_helper(t, self.manip_bases)
		return embed_dict
		

class PosEncodingENeRV(PosEncoding):
	def __init__(self, pe_lbase, pe_levels, xy_lbase, xy_levels, manip_lbase, manip_levels, fc_h, fc_w):
		super().__init__()
		self.pe_bases = pe_lbase ** torch.arange(int(pe_levels)) * math.pi
		self.xy_bases = xy_lbase ** torch.arange(int(xy_levels)) * math.pi
		self.manip_bases = manip_lbase ** torch.arange(int(manip_levels)) * math.pi
		self.fc_h = fc_h
		self.fc_w = fc_w

	def pos_embed_helper(self, pos, bases):
		pos = pos[:,None]
		value_list = pos * bases.to(pos.device)
		return torch.cat([torch.sin(value_list), torch.cos(value_list)], dim=-1)

	def pos_embed_helper_patches(self, pos, bases):
		pos = pos.unsqueeze(2)
		bases_batch = einops.repeat(bases, 'd -> b 1 d', b=pos.shape[0])
		value_list = torch.bmm(pos, bases_batch.to(pos.device))
		return torch.cat([torch.sin(value_list), torch.cos(value_list)], dim=-1)

	def forward(self, t, idx_max=None):
		if idx_max is None:
			assert t.dim() == 1
		embed_dict = {}

		if idx_max is None:
			xy_coord = torch.stack( 
				torch.meshgrid(
					torch.arange(self.fc_h) / self.fc_h, torch.arange(self.fc_w) / self.fc_w
				), dim=0
			).flatten(1, 2).to(t.device)
			x_embed = self.pos_embed_helper(xy_coord[0], self.xy_bases)
			y_embed = self.pos_embed_helper(xy_coord[1], self.xy_bases)
			xy_embed = torch.cat([x_embed, y_embed], dim=1)
			embed_dict['xy_embed'] = xy_embed
		else:
			xy_coord = torch.stack( 
				torch.meshgrid(
					torch.arange(self.fc_h*idx_max[1]) / (self.fc_h*idx_max[1]), torch.arange(self.fc_w*idx_max[2]) / (self.fc_w*idx_max[2])
				), dim=0
			).to(t.device)
			xy_coords = []
			for cur_t in t:
				xy_coords.append(xy_coord[:,self.fc_h*cur_t[1]:self.fc_h*cur_t[1]+self.fc_h,self.fc_w*cur_t[2]:self.fc_w*cur_t[2]+self.fc_w].contiguous().flatten(1, 2))
			xy_coord = torch.stack(xy_coords, dim=0)

			x_embed = self.pos_embed_helper_patches(xy_coord[:,0], self.xy_bases)
			y_embed = self.pos_embed_helper_patches(xy_coord[:,1], self.xy_bases)
			xy_embed = torch.cat([x_embed, y_embed], dim=2)
			embed_dict['xy_embed'] = xy_embed

			t = (t[:,0]*idx_max[1]*idx_max[2] + \
                        t[:,1]*idx_max[2] + \
                        t[:,2]) / \
                        (idx_max[0] * idx_max[1] * idx_max[2])
	
		embed_dict['t_embed'] = self.pos_embed_helper(t, self.pe_bases).view(t.size(0), -1, 1, 1).float()

		embed_dict['t_manipulate_embed'] = self.pos_embed_helper(t, self.manip_bases)

		return embed_dict


class PosEncodingHiENeRV(PosEncoding):
	def __init__(self, size, channels, encoding_config):
		super().__init__()
		self.size = size
		self.channels = channels

		T, H, W = self.size
		C = self.channels

		# Grids
		self.grids = nn.ParameterList()
		self.grid_expands = nn.ModuleList()

		self.grid_level = encoding_config['base_grid_level']
		self.grid_sizes = []

		T_grid, H_grid, W_grid, C_grid = encoding_config['base_grid_size']
		T_scale, H_scale, W_scale, C_scale = encoding_config['base_grid_level_scale']

		for i in range(self.grid_level):
			T_i, H_i, W_i, C_i = int(T_grid / T_scale ** i), int(H_grid / H_scale ** i), int(W_grid / W_scale ** i), int(C_grid / C_scale ** i)
			self.grid_sizes.append((T_i, H_i, W_i, C_i))
			self.grids.append(FeatureGrid((T_i * H_i * W_i, C_i), init_scale=encoding_config['base_grid_init_scale']))
			self.grid_expands.append(GridTrilinear3D((T, H, W)))

		self.apply(self._init_weights)
		self.manip_bases = encoding_config['manip_lbase'] ** torch.arange(int(encoding_config['manip_levels'])) * math.pi

	def pos_embed_helper(self, pos, bases):
		pos = pos[:,None]
		value_list = pos * bases.to(pos.device)
		return torch.cat([torch.sin(value_list), torch.cos(value_list)], dim=-1)

	def _init_weights(self, m):
		if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
			if m.bias is not None:
				nn.init.zeros_(m.bias)

	def extra_repr(self):
		s = 'size={size}, channels={channels}, grid_level={grid_level}, grid_sizes={grid_sizes}'
		return s.format(**self.__dict__)

	def forward(self, idx: torch.IntTensor, idx_max: tuple[int, int, int], padding: tuple[int, int, int]):
		"""
		Inputs:
			idx: patch index tensor with shape [N, 3]
			idx_max: list of 3 ints. Represents the range of patch indexes.
			patch_mode: if True, the input is a patch from the full video, and the faster implementation will be used.

		Output:
			a tensor with shape [N, T, H, W, C]
		"""
		assert idx.ndim == 2 and idx.shape[1] == 3
		assert len(idx_max) == 3

		# Compute the global voxels coordinates
		patch_size = tuple(self.size[d] // idx_max[d] for d in range(3))
		patch_padded_size = tuple(patch_size[d] + 2 * padding[d] for d in range(3))

		px_idx, px_mask = compute_pixel_idx_3d(idx, idx_max, self.size, padding=padding, clipped=True)
		px_idx_flat = (px_idx[0][:, :, None, None] * self.size[1] * self.size[2]
						+ px_idx[1][:, None, :, None] * self.size[2]
						+ px_idx[2][:, None, None, :]).view(-1)
		px_mask_flat = (px_mask[0][:, :, None, None, None]
						* px_mask[1][:, None, :, None, None] 
						* px_mask[2][:, None, None, :, None]).view(-1, 1)

		# Encode
		enc_splits = [self.grid_expands[i](self.grids[i]().view(self.grid_sizes[i])) for i in range(self.grid_level)]
		enc = torch.concat(enc_splits, dim=-1)

		output = (px_mask_flat * torch.index_select(enc.view(self.size[0] * self.size[1] * self.size[2], self.channels), 0, px_idx_flat)) \
					.view((idx.shape[0],) + patch_padded_size + (self.channels,))
		assert tuple(output.shape) == tuple((idx.shape[0],) + patch_padded_size + (self.channels,)), f'shape: {output.shape}, expected: {idx.shape[0]}, {patch_padded_size}, {self.channels}'

		embed_dict = {
			't_embed': output
		}
		embed_dict['t_manipulate_embed'] = self.pos_embed_helper(idx[:,0], self.manip_bases)

		return embed_dict
	

class PosEncodingHENeRV(PosEncoding):
	def __init__(self, enc_strds, enc_dims, xy_lbase, xy_levels, manip_lbase, manip_levels, fc_h, fc_w):
		super().__init__()
		self.xy_bases = xy_lbase ** torch.arange(int(xy_levels)) * math.pi
		self.manip_bases = manip_lbase ** torch.arange(int(manip_levels)) * math.pi
		self.fc_h = fc_h
		self.fc_w = fc_w

		enc_dim1, enc_dim2 = [int(x) for x in enc_dims.split('_')]
		c_out_list = [enc_dim1] * len(enc_strds)
		c_out_list[-1] = enc_dim2
		self.encoder = ConvNeXt(stage_blocks=1, strds=enc_strds, dims=c_out_list,
			drop_path_rate=0)

	def pos_embed_helper(self, pos, bases):
		pos = pos[:,None]
		value_list = pos * bases.to(pos.device)
		return torch.cat([torch.sin(value_list), torch.cos(value_list)], dim=-1)

	def forward(self, image, t, qbit=None):
		assert t.dim() == 1
		embed_dict = {}

		xy_coord = torch.stack( 
			torch.meshgrid(
				torch.arange(self.fc_h) / self.fc_h, torch.arange(self.fc_w) / self.fc_w
			), dim=0
		).flatten(1, 2).to(t.device)
	
		encoded_embed = self.encoder(image).view(t.size(0), -1, 1, 1)
		if qbit is not None:
			#encoded_embed = encoded_embed#.half().float()
			axis = compute_best_quant_axis(encoded_embed)
			quant_embed, quant_scale = _quantize_ste(encoded_embed, qbit, axis)
			quant_embed = quant_embed.to(torch.int32)
			encoded_embed = quant_embed.float() * quant_scale.to(quant_embed.device).float()
		embed_dict['t_embed'] = encoded_embed

		x_embed = self.pos_embed_helper(xy_coord[0], self.xy_bases)
		y_embed = self.pos_embed_helper(xy_coord[1], self.xy_bases)
		xy_embed = torch.cat([x_embed, y_embed], dim=1)
		embed_dict['xy_embed'] = xy_embed

		embed_dict['t_manipulate_embed'] = self.pos_embed_helper(t, self.manip_bases)

		return embed_dict


class PosEncodingHNeRV(PosEncoding):
	def __init__(self, enc_strds, enc_dims, modelsize, frame_hw, full_data_length):
		super().__init__()

		enc_dim1, embed_ratio = [int(x) for x in enc_dims.split('_')]
		embed_hw = np.prod(frame_hw) / np.prod(enc_strds)**2
		enc_dim2 = int(embed_ratio * modelsize * 1e6 / full_data_length / embed_hw) if embed_ratio < 1 else int(embed_ratio) 
		c_out_list = [enc_dim1] * len(enc_strds)
		c_out_list[-1] = enc_dim2
		self.encoder = ConvNeXt(stage_blocks=1, strds=enc_strds, dims=c_out_list,
			drop_path_rate=0)

	def forward(self, image, qbit=None):
		encoded_embed = self.encoder(image)
		if qbit is not None:
			#encoded_embed = encoded_embed#.half().float()
			axis = compute_best_quant_axis(encoded_embed)
			quant_embed, quant_scale = _quantize_ste(encoded_embed, qbit, axis)
			quant_embed = quant_embed.to(torch.int32)
			encoded_embed = quant_embed.float() * quant_scale.to(quant_embed.device).float()
		return {'t_embed': encoded_embed}


class PosEncodingDiffNeRV(PosEncoding):
	def __init__(self, enc_strds, diff_enc_list, c1_dim, d_dim, c2_dim):
		super().__init__()

		self.encoder_layers = nn.ModuleList()
		self.diff_enc_layers = nn.ModuleList()

		for k, stride in enumerate(enc_strds):
			if k == 0:
				c0 = 3
			else:
				c0 = c1_dim

			self.encoder_layers.append(nn.Conv2d(c0, c1_dim, kernel_size=stride, stride=stride))
			self.encoder_layers.append(LayerNorm(c1_dim, eps=1e-6, data_format="channels_first"))
			self.encoder_layers.append(ConvNeXtBlock(dim_in=c1_dim, dim_hidden=4*c1_dim, dim_out=c1_dim))
			self.enc_embedding_layer = nn.Conv2d(c1_dim, d_dim, kernel_size=1, stride=1)

			if k<len(diff_enc_list):
				if k == 0:
					c0 = 6
				else:
					c0 = c1_dim
				self.diff_enc_layers.append(nn.Conv2d(c0, c1_dim, kernel_size=diff_enc_list[k],
						stride=diff_enc_list[k]))
				self.diff_enc_layers.append(LayerNorm(c1_dim, eps=1e-6, data_format="channels_first"))
				self.diff_enc_layers.append(ConvNeXtBlock(dim_in=c1_dim, dim_hidden=4*c1_dim, dim_out=c1_dim))

		self.diff_enc_ebd_layer = nn.Conv2d(c1_dim, 2, kernel_size=1, stride=1)
		self.enc_c2_layer = nn.Conv2d(d_dim, c2_dim, kernel_size=1, stride=1)

	def forward(self, data):
		content_embedding = data[:, 1, :, :, :]
		content_p = data[:, 0, :, :, :]
		content_f = data[:, 2, :, :, :]
		content_gt = data[:, 1, :, :, :]

		for encoder_layer in self.encoder_layers:
			content_embedding = encoder_layer(content_embedding)
		cnt_output   = self.enc_embedding_layer(content_embedding)

		diff_p = content_gt - content_p
		diff_f = content_f - content_gt
		diff = torch.stack([diff_p, diff_f], dim=2)
		diff = diff.view(diff.size(0),-1,diff.size(-2),diff.size(-1))
		for diff_enc_layer in self.diff_enc_layers:
			diff = diff_enc_layer(diff)

		diff = self.diff_enc_ebd_layer(diff) 
		output = self.enc_c2_layer(cnt_output)

		return {
			"diff": diff,
			"output": output
		}
	
# positional encoder for keyframes
class PosEncodingDivNeRVKeyFrames(PosEncoding):
	def __init__(self, kernel_size=3, stride=1, stride_list=[], bias=True):
		super().__init__()
		n_resblocks = len(stride_list)

		# define head module
		m_head = nn.Sequential(
			nn.Conv3d(3, 64, kernel_size=(1, kernel_size, kernel_size), stride=(1, stride, stride), padding=(0, kernel_size//2, kernel_size//2), bias=bias),
			nn.GELU(),
		)
		m_body = []
		for i in range(n_resblocks):
			m_body.append(nn.Sequential(
							nn.Conv3d(64, 64, kernel_size=(1, stride_list[i], stride_list[i]), stride=(1, stride_list[i], stride_list[i]), padding=(0, 0, 0), bias=bias),
							nn.GELU(),
							)
						)
		
		self.head = nn.Sequential(*m_head)
		self.body = nn.ModuleList(m_body)

	def forward(self, x):
		key_feature_list = [x]
		x = self.head(x)
		for stage in self.body:
			x = stage(x)
			key_feature_list.append(x)
		return key_feature_list[::-1]


# positional encoder for frame index
class DivNeRVFrameIndexPositionalEncoding(PosEncoding):
    def __init__(self, pe_embed):
        super(DivNeRVFrameIndexPositionalEncoding, self).__init__()
        self.pe_embed = pe_embed.lower()
        if self.pe_embed == 'none':
            self.embed_length = 1
        else:
            self.lbase, self.levels = [float(x) for x in pe_embed.split('_')]
            self.levels = int(self.levels)
            self.embed_length = 2 * self.levels

    def __repr__(self):
        return f"Positional Encoder: pos_b={self.lbase}, pos_l={self.levels}, embed_length={self.embed_length}, to_embed={self.pe_embed}"

    def forward(self, pos):
        if self.pe_embed == 'none':
            return pos[:,None]
        else:
            pe_list = []
            for i in range(self.levels):
                temp_value = pos * self.lbase ** (i) * math.pi
                pe_list += [torch.sin(temp_value), torch.cos(temp_value)]
            result = torch.stack(pe_list, 1)
            return result

class PosEncodingFFNeRV(PosEncoding):
	def __init__(self, t_dim, fc_dim, fc_h, fc_w, wbit, manip_lbase=None, manip_levels=None):
		super().__init__()
		#self.quantize_fn = weight_quantize_fn(wbit)
		self.video_grid = nn.ParameterList()
		for t in t_dim:
			self.video_grid.append(nn.Parameter(nn.init.xavier_uniform_(torch.empty(t,fc_dim//len(t_dim),fc_h,fc_w))))

		if manip_lbase is not None and manip_levels is not None:
			self.manip_bases = manip_lbase ** torch.arange(int(manip_levels)) * math.pi
		else:
			self.manip_bases = None

	def pos_embed_helper(self, pos, bases):
		pos = pos[:,None]
		value_list = pos * bases.to(pos.device)
		return torch.cat([torch.sin(value_list), torch.cos(value_list)], dim=-1)

	def forward(self, t):
		assert t.dim() == 1

		out_list = []
		for param in self.video_grid: # multi-resolution grids
			#vg = self.quantize_fn(param)
			vg = param
			# interpolate grid features
			inp = t*(param.size(0))
			left = torch.floor(inp+1e-6).long()
			right = torch.min(left+1, torch.tensor(param.size(0)-1))
			d_left = (inp - left).view(-1, 1, 1, 1)
			d_right = (right - inp).view(-1, 1, 1, 1)
			out_list.append(d_right*vg[left] + d_left*vg[right] - ((right-left-1).view(-1,1,1,1))*vg[left])
		output = out_list[0]
		# concat latent features from multi-resolution grids
		for i in range(len(out_list)-1):
			output = torch.cat([output, out_list[i+1]],dim=1)

		embed_dict = {'t_embed': output}

		if self.manip_bases is not None:
			embed_dict['t_manipulate_embed'] = self.pos_embed_helper(t, self.manip_bases)

		return embed_dict


class GridEncodingBase(nn.Module):
    """
    K: Kernel size [K_t, K_h, K_w].
    C: Number of output channels.
    grid_size: First level grid's size.
    """
    def __init__(self, K=(1, 2, 2), C=128, grid_size=[120, 9, 16, 64], grid_level=3, grid_level_scale=[2., 1., 1., .5], init_scale=1e-3, align_corners=True):
        super().__init__()
        self.K = K
        self.C = C
        self.grid_sizes = []
        self.grid_level = grid_level
        self.grid_level_scale = grid_level_scale
        self.init_scale = init_scale
        self.align_corners = align_corners

        # Weights (saved in 2D to prevent converted by .to(channels_last))
        self.grids = nn.ModuleList()
        for i in range(self.grid_level):
            T_grid_i, H_grid_i, W_grid_i, C_grid_i = tuple((int(grid_size[d] / (self.grid_level_scale[d] ** i)) for d in range(4)))
            self.grid_sizes.append((T_grid_i, H_grid_i, W_grid_i, self.K[0], self.K[1], self.K[2], C_grid_i))
            self.grids.append(FeatureGrid((T_grid_i * H_grid_i * W_grid_i, self.K[0] * self.K[1] * self.K[2] * C_grid_i), init_scale=self.init_scale))

        # Linear
        self.linear = nn.Linear(sum([C_em for _, _, _, _, _, _, C_em in self.grid_sizes]), self.C)

    def extra_repr(self):
        s = 'C={C}, grid_sizes={grid_sizes}, grid_level={grid_level}, grid_level_scale={grid_level_scale}, '\
            'init_scale={init_scale}, align_corners={align_corners}'
        return s.format(**self.__dict__)

    def forward(self, coor_t: torch.FloatTensor, coor_h: Optional[torch.FloatTensor]=None, coor_w: Optional[torch.FloatTensor]=None):
        """        
        Inputs:
            coor_t/coor_h/coor_w: coordinates with shape [N, T/H/W]
        Output:
            a tensor with shape [N, T, H, W, K_t * K_h * K_w, C]
        """
        assert coor_t.ndim == 2 and (coor_h is None or coor_h.ndim == 2) and (coor_w is None or coor_w.ndim == 2)
        assert (coor_h is None) == (coor_w is None)
        N, T = coor_t.shape
        H = coor_h.shape[1] if coor_h is not None else 1
        W = coor_w.shape[1] if coor_w is not None else 1

        # Sampling coordinates
        if H > 1 and W > 1:
            coor_t = coor_t[:, :, None, None, None].expand(N, T, H, W, 1)
            coor_h = coor_h[:, None, :, None, None].expand(N, T, H, W, 1)
            coor_w = coor_w[:, None, None, :, None].expand(N, T, H, W, 1)
            coor_wht = torch.cat([coor_w, coor_h, coor_t], dim=-1) # (x, y, t)/(w, h, t) order
        else:
            coor_t = coor_t.view(N, T, 1, 1, 1)
            coor_wht = F.pad(coor_t, (2, 0)) # (x, y, t)/(w, h, t) order

        enc = []

        for i in range(self.grid_level):
            T_grid_i, H_grid_i, W_grid_i, K0, K1, K2, C_grid_i = self.grid_sizes[i]

            # Interpolate in all dimenions
            weight_i = self.grids[i]().view(T_grid_i, H_grid_i, W_grid_i, K0 * K1 * K2 * C_grid_i)
            enc_i = interpolate3D(grid=weight_i, coor_wht=coor_wht, align_corners=self.align_corners)
            enc.append(enc_i.contiguous().view(N, T * H * W * K0 * K1 * K2, C_grid_i))

        return self.linear(torch.concat(enc, dim=-1)).view(N, T, H, W, math.prod(self.K) * self.C)


class TemporalLocalGridEncoding(GridEncodingBase):
    def __init__(self, K=(1, 2, 2), C=128, grid_size=[10, 64], grid_level=3, grid_level_scale=[1., 1.], init_scale=1e-3, align_corners=True):
        super().__init__(K=K, C=C, grid_size=(grid_size[0], 1, 1, grid_size[1]), grid_level=grid_level, 
                         grid_level_scale=(grid_level_scale[0], 1, 1, grid_level_scale[1]), 
                         init_scale=init_scale, align_corners=align_corners)

    def forward(self, coor_t: torch.FloatTensor, coor_h: Optional[torch.FloatTensor]=None, coor_w: Optional[torch.FloatTensor]=None):
        assert coor_t.ndim == 2 and (coor_h is None or coor_h.ndim == 2) and (coor_w is None or coor_w.ndim == 2)
        assert (coor_h is None) == (coor_w is None)
        N, T = coor_t.shape
        H = coor_h.shape[1] if coor_h is not None else 1
        W = coor_w.shape[1] if coor_w is not None else 1
        return super().forward(coor_t, coor_h, coor_w).view(N, T, H, W, self.K[0], self.K[1], self.K[2], self.C)


class PosEncodingHiNeRVLocal(PosEncoding):
	def __init__(self, scale, channels, cfg):
		super().__init__()
		self.scale = scale
		self.channels = channels

		self.coor_type = 'normalized'
		self.enc_type = 'temp_local_grid'
		self.coor = NormalizedCoordinate(align_corners=False)

		self.enc = TemporalLocalGridEncoding(K=self.scale, C=self.channels, grid_size=cfg['grid_size'], grid_level=cfg['grid_level'], grid_level_scale=cfg['grid_level_scale'],
													init_scale=cfg['grid_init_scale'], align_corners=cfg['align_corners'])
		self.f_type = 'temp_local'

	def extra_repr(self):
		s = 'scale={scale}, channels={channels}, coor_type={coor_type}, enc_type={enc_type}'
		return s.format(**self.__dict__)

	def compute_temp_local_encoding(self, x: torch.Tensor, idx: torch.IntTensor, idx_max: tuple[int, int, int],
									size: tuple[int, int, int], scale: tuple[int, int, int], padding: tuple[int, int, int]):
		N, T, H, W, C = x.shape

		# This implementation is only correct with no temporal upsampling
		assert scale[0] == 1

		# Compute the global voxel coordinates before upscaling
		pre_size = (size[0] // scale[0],)
		pre_padding = (int(math.ceil(padding[0] / scale[0])),)
		pre_idx = compute_pixel_idx_1d(idx[:, 0:1], idx_max[0:1], pre_size, pre_padding, clipped=False, return_mask=False)
		coor_t = self.coor(pre_idx[0], pre_size[0])
		#print(f'coor_t shape: {coor_t.shape}')

		# Compute the local voxel indexes
		px_idx, px_mask = compute_pixel_idx_3d(idx, idx_max, size, padding, clipped=False, return_mask=True)
		px_mask_3d = (px_mask[0][:, :, None, None, None]
						* px_mask[1][:, None, :, None, None]
						* px_mask[2][:, None, None, :, None])
		lpx_idx = tuple(px_idx[d] % scale[d] for d in range(3))

		# Compute the encoding indexes
		enc_idx = tuple(torch.arange(scale[d], device=x.device) for d in range(3))

		# Encoding
		M = tuple(lpx_idx[d][:, :, None] == enc_idx[d][None, None, :] for d in range(3))
		M_3d = (M[0][:, :, None, None, :, None, None] *
				M[1][:, None, :, None, None, :, None] *
				M[2][:, None, None, :, None, None, :])
		local_enc = self.enc(coor_t)
		#print(f'local_enc shape: {local_enc.shape}')
		#print(M_3d.shape)
		#print(N, T, H, W, C)
		#print(N, T, H * W, scale[0] * scale[1] * scale[2])
		local_enc_masked = px_mask_3d * torch.matmul(M_3d.view(N, T, H * W, scale[0] * scale[1] * scale[2]).to(local_enc.dtype),
														local_enc.view(N, T, scale[0] * scale[1] * scale[2], C)).view_as(x)

		return x + local_enc_masked

	def forward(self, x: Optional[torch.Tensor], idx: torch.IntTensor, idx_max: tuple[int, int, int],
				size: tuple[int, int, int], scale: tuple[int, int, int], padding: tuple[int, int, int]):
		""" 
		Inputs:
			x: input tensor with shape [N, T1, H1, W1, C]
			idx: patch index tensor with shape [N, 3]
			idx_max: list of 3 ints. Represents the range of patch indexes.
			size: list of 3 ints. Represents the size of the fulle video. It does not have to be the same as the input size, as the input can be a patch from the full video.
			scale: list of 3 ints. Represents the scale factor. This will be used to compute the output size.
			padding: list of 3 ints. Represents the padding size. This will be used to compute the output size.
		Outputs:
			a tensor with shape [N, T2, H2, W2, C]
		"""
		assert x is None or x.ndim == 5, x.shape
		assert idx.ndim == 2 and idx.shape[1] == 3, idx.shape
		assert len(idx_max) == 3
		assert len(scale) == 3
		assert len(size) == 3
		assert len(padding) == 3

		if x is None:
			x = torch.zeros((1,) + tuple(size[d] // idx_max[d] + 2 * padding[d] for d in range(3)) + (1,), device=idx.device)

		if self.f_type == 'temp_local':
			x = self.compute_temp_local_encoding(x, idx, idx_max, size, scale, padding)
		else:
			raise NotImplementedError
		
		#print(f'hinerv local grid output shape: {x.shape}')

		return x


class PosEncodingHiNeRV(PosEncoding):
	def __init__(self, size, channels, encoding_config):
		super().__init__()
		self.size = size
		self.channels = channels

		T, H, W = self.size
		C = self.channels

		# Grids
		self.grids = nn.ParameterList()
		self.grid_expands = nn.ModuleList()

		self.grid_level = encoding_config['base_grid_level']
		self.grid_sizes = []

		T_grid, H_grid, W_grid, C_grid = encoding_config['base_grid_size']
		T_scale, H_scale, W_scale, C_scale = encoding_config['base_grid_level_scale']

		for i in range(self.grid_level):
			T_i, H_i, W_i, C_i = int(T_grid / T_scale ** i), int(H_grid / H_scale ** i), int(W_grid / W_scale ** i), int(C_grid / C_scale ** i)
			self.grid_sizes.append((T_i, H_i, W_i, C_i))
			self.grids.append(FeatureGrid((T_i * H_i * W_i, C_i), init_scale=encoding_config['base_grid_init_scale']))
			self.grid_expands.append(GridTrilinear3D((T, H, W)))

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
			if m.bias is not None:
				nn.init.zeros_(m.bias)

	def extra_repr(self):
		s = 'size={size}, channels={channels}, grid_level={grid_level}, grid_sizes={grid_sizes}'
		return s.format(**self.__dict__)

	def forward(self, idx: torch.IntTensor, idx_max: tuple[int, int, int], padding: tuple[int, int, int]):
		"""
		Inputs:
			idx: patch index tensor with shape [N, 3]
			idx_max: list of 3 ints. Represents the range of patch indexes.
			patch_mode: if True, the input is a patch from the full video, and the faster implementation will be used.

		Output:
			a tensor with shape [N, T, H, W, C]
		"""
		assert idx.ndim == 2 and idx.shape[1] == 3
		assert len(idx_max) == 3

		#print(f'idx {idx}, idx_max {idx_max}, size {self.size}, padding {padding}')

		# Compute the global voxels coordinates
		patch_size = tuple(self.size[d] // idx_max[d] for d in range(3))
		patch_padded_size = tuple(patch_size[d] + 2 * padding[d] for d in range(3))

		#print(f'patch_size {patch_size}, patch_padded_size {patch_padded_size}')

		px_idx, px_mask = compute_pixel_idx_3d(idx, idx_max, self.size, padding=padding, clipped=True)
		#print(f'px_idx {px_idx}')
		#print(f'px_mask {px_mask}')
		px_idx_flat = (px_idx[0][:, :, None, None] * self.size[1] * self.size[2]
						+ px_idx[1][:, None, :, None] * self.size[2]
						+ px_idx[2][:, None, None, :]).view(-1)
		px_mask_flat = (px_mask[0][:, :, None, None, None]
						* px_mask[1][:, None, :, None, None] 
						* px_mask[2][:, None, None, :, None]).view(-1, 1)

		# Encode
		enc_splits = [self.grid_expands[i](self.grids[i]().view(self.grid_sizes[i])) for i in range(self.grid_level)]
		enc = torch.concat(enc_splits, dim=-1)
		
		#print(f'enc shape {enc.shape}')
		#print(f'px_idx_flat {px_idx_flat}')
		#print(f'px_mask_flat {px_mask_flat}')
		#print(f'enc reshape {self.size[0] * self.size[1] * self.size[2]}, {self.channels}')

		output = (px_mask_flat * torch.index_select(enc.view(self.size[0] * self.size[1] * self.size[2], self.channels), 0, px_idx_flat)) \
					.view((idx.shape[0],) + patch_padded_size + (self.channels,))
		assert tuple(output.shape) == tuple((idx.shape[0],) + patch_padded_size + (self.channels,)), f'shape: {output.shape}, expected: {idx.shape[0]}, {patch_padded_size}, {self.channels}'

		#print(f'hinerv grid output shape: {output.shape}')

		return {'t_embed': output}


class PosEncodingNeRF(PosEncoding):
	'''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
	def __init__(self, dim_in, sidelength=None, fn_samples=None, use_nyquist=True,num_freq=None,include_coord=True,freq_last=False):
		super().__init__()

		self.dim_in = dim_in
		self.include_coord = include_coord
		self.freq_last = freq_last

		if self.dim_in == 3:
			self.num_frequencies = 10
		elif self.dim_in == 2:
			assert sidelength is not None
			if isinstance(sidelength, int):
				sidelength = (sidelength, sidelength)
			self.num_frequencies = 4
			if use_nyquist and num_freq is None:
				self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
			elif num_freq is not None:
				self.num_frequencies = num_freq
			print('Num Frequencies: ',self.num_frequencies)
		elif self.dim_in == 1:
			assert fn_samples is not None
			self.num_frequencies = 4
			if use_nyquist:
				self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)

		self.output_dim = dim_in + 2 * dim_in * self.num_frequencies

		#Fixed. Not trainable. 
		self.freq_bands = nn.parameter.Parameter(2**torch.arange(self.num_frequencies) * np.pi, requires_grad=False)


	def get_num_frequencies_nyquist(self, samples):
		nyquist_rate = 1 / (2 * (2 * 1 / samples))
		return int(math.floor(math.log(nyquist_rate, 2)))


	def forward(self, coords, single_channel=False):
		
		if single_channel:
			in_features = coords.shape[-1]
		else:
			in_features = self.in_features

		#removes for loop over sine and cosine.
		#bad, but optimal code. lifted from https://github.com/nexuslrf/CoordX/blob/main/modules.py
		coords_pos_enc = coords.unsqueeze(-2) * self.freq_bands.reshape([1]*(len(coords.shape)-1) + [-1, 1]) #2*pi*coord
		sin = torch.sin(coords_pos_enc)
		cos = torch.cos(coords_pos_enc)

		coords_pos_enc = torch.cat([sin, cos], -1).reshape(list(coords_pos_enc.shape)[:-2] + [-1])
		
		if self.include_coord:
			coords_pos_enc = torch.cat([coords, coords_pos_enc], -1)

		if self.freq_last:
			sh = coords_pos_enc.shape[:-1]
			coords_pos_enc = coords_pos_enc.reshape(*sh, -1, in_features).transpose(-1,-2).reshape(*sh, -1)

		
		return coords_pos_enc


class PosEncodingGaussian(PosEncoding):
	"""
	An implementation of Gaussian Fourier feature mapping.

	"Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
	   https://arxiv.org/abs/2006.10739
	   https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

	Given an input of size [batches, num_input_channels, width, height],
	 returns a tensor of size [batches, mapping_size*2, width, height].
	"""

	def __init__(self, dim_in, mapping_size=256, scale=10):
		super().__init__()

		self._num_input_channels = dim_in
		self._mapping_size = mapping_size
		self._B = torch.randn((dim_in, mapping_size)) * scale
		self.output_dim = mapping_size*2

	def forward(self, x):
		assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

		batches, channels, width, height = x.shape

		assert channels == self._num_input_channels,\
			"Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

		# Make shape compatible for matmul with _B.
		# From [B, C, W, H] to [(B*W*H), C].
		x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

		x = x @ self._B.to(x.device)

		# From [(B*W*H), C] to [B, W, H, C]
		x = x.view(batches, width, height, self._mapping_size)
		# From [B, W, H, C] to [B, C, W, H]
		x = x.permute(0, 3, 1, 2)

		x = 2 * np.pi * x
		return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


class AdaIN(nn.Module):
	def __init__(self):
		super().__init__()

	def mu(self, x):
		""" Takes a (n,c,h,w) tensor as input and returns the average across
		it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
		return torch.sum(x,(2,3))/(x.shape[2]*x.shape[3])

	def sigma(self, x):
		""" Takes a (n,c,h,w) tensor as input and returns the standard deviation
		across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
		the permutations are required for broadcasting"""
		return torch.sqrt((torch.sum((x.permute([2,3,0,1])-self.mu(x)).permute([2,3,0,1])**2,(2,3))+0.000000023)/(x.shape[2]*x.shape[3]))

	def forward(self,feature,target_mu,target_sigma):
		"""
			Feature is (N,C,H,W)
		"""
		
		feature_mu = self.mu(feature)
		feature_sigma = self.sigma(feature)
		
		return (target_sigma*((feature.permute([2,3,0,1])-feature_mu)/feature_sigma) + target_mu).permute([2,3,0,1])


