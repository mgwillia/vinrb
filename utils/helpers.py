### code adapted from https://github.com/abhyantrika/mediainr ###

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math

import numpy as np
import os,glob
import pickle
import compress_pickle

from pytorch_msssim import ms_ssim


def psnr(img1, img2):
    """Calculates PSNR between two images.

    Args:
        img1 (torch.Tensor):
        img2 (torch.Tensor):
    """
    return 20. * np.log10(1.) - 10. * (img1 - img2).detach().pow(2).mean().log10().to('cpu').item()


def ssim(img1, img2):
    return ms_ssim(img1.unsqueeze(0).float().detach(), img2.unsqueeze(0).detach(), data_range=1, size_average=False).to('cpu').item()


def get_padded_patch_size(tensor_shape:list,patch_size:int):
    """
    Get the size of the padded tensor when patchified.

    Args:
        tensor_shape (tuple): Shape of the input tensor. C,H,W
        patch_size (int): Size of the patch.
    Returns:
        tuple: Shape of the padded tensor. C,H,W
    """
    channels, height, width = tensor_shape
    pad_height = (patch_size - height % patch_size) % patch_size
    pad_width = (patch_size - width % patch_size) % patch_size
    return channels,height+pad_height,width+pad_width


def save_tensor_img(tensor, filename:str='temp.png'):
    """
        conver to image and save.
    """	
    #convert tensor to int8 pytorch
    tensor = (tensor * 255).clamp(0, 255).to(torch.uint8).squeeze()
    tensor = tensor.cpu().detach()
    torchvision.io.write_png(tensor, filename)    


def custom_collate_fn(batch:dict):
    # Assuming each item in the batch is a dictionary
    # We initialize an empty batch dictionary
    collated_batch = {}

    # Stack the data, frame_ids, and paths
    collated_batch['features'] = torch.stack([item['features'] for item in batch])
    collated_batch['frame_ids'] = torch.tensor([item['frame_ids'] for item in batch])
    collated_batch['group_id'] = torch.tensor([item['group_id'] for item in batch])
    collated_batch['norm_idx'] = torch.tensor([item['norm_idx'] for item in batch])
    if 'coordinates' in batch[0]:
        collated_batch['coordinates'] = torch.stack([item['coordinates'] for item in batch])
    if 'thw_idx' in batch[0]:
        collated_batch['thw_idx'] = torch.stack([item['thw_idx'] for item in batch])

    return collated_batch

def div_nerv_collate_fn(batch):
    batched_output_list = []
    for i in range(len(batch[0])):
        if torch.is_tensor(batch[0][i]):
            batched_output = torch.stack([single_batch[i] for single_batch in batch], dim=0)
        elif type(batch[0][i]) is dict:
            batched_output = {}
            for k, v in batch[0][i].items():
                batched_output[k] = torch.stack([single_batch[i][k] for single_batch in batch], dim=0)
        batched_output_list.append(batched_output)
    return batched_output_list

def nvp_collate_fn(batch:dict):
    # Assuming each item in the batch is a dictionary
    # We initialize an empty batch dictionary
    collated_batch = {}

    # Stack the data, frame_ids, and paths
    collated_batch['features'] = torch.stack([item['features'] for item in batch])
    collated_batch['frame_ids'] = torch.tensor([item['frame_ids'] for item in batch])
    collated_batch['coordinates'] = torch.stack([item['coordinates'] for item in batch])
    collated_batch['temporal_steps'] = torch.stack([item['temporal_steps'] for item in batch])

    return collated_batch

def adjust_lr_linear_warmup_cosine_annealing(optimizer, cur_epoch, cur_iter, data_size, lr, warmup, epochs):
    # Adjust the learning rate using linear warmup followed by cosine annealing.
    cur_epoch = cur_epoch + (float(cur_iter) / data_size)
    if cur_epoch < warmup:
        lr_mult = 0.1 + 0.9 * cur_epoch / warmup
    else:
        lr_mult = 0.5 * (math.cos(math.pi * (cur_epoch - warmup)/ (epochs - warmup)) + 1.0)

    lr = lr * lr_mult
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr
    return lr


def make_dir(path:str):
    if not os.path.exists(path):
        os.makedirs(path,exist_ok=True)

def save_pickle(data,filename:str,compressed:bool=False):
    with open(filename, 'wb') as f:
        if compressed:
            compress_pickle.dump(data, f,compression='lzma',set_default_extension=False,pickler_method='optimized_pickle')
        else:
            pickle.dump(data, f,protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filename:str,compressed:bool=False):
    with open(filename, 'rb') as f:
        if compressed:
            return compress_pickle.load(f,compression='lzma')
        return pickle.load(f)

def find_ckpt(save_dir:str):
    """
        Function to recursively find ckpt files in the directroy 
        return the path of the latest one. 
    """
    ckpt_files = glob.glob(save_dir+'/**/*.ckpt',recursive=True)
    #ckpt_files = sorted(ckpt_files,key=lambda x: int(x.split('_')[-1].split('.')[0]))
    #sort accoriding to creation time. 
    ckpt_files = sorted(ckpt_files,key=os.path.getctime)
    if ckpt_files == []:
        return None
    return ckpt_files[-1]

def compute_nerv_dim(modelsize:int, reduce:int, ks1:int, ks2:int, dec_strds:list, lower_width:int, embed_param:int, embed_dim:int, fc_param:int):

    decoder_size = modelsize * 1e6 - embed_param
    ch_reduce = 1. / reduce
    fix_ch_stages = len(dec_strds)
    a =  ch_reduce * sum([ch_reduce**(2*i) * (s**2 if type(s) == int else s[0] * s[1]) * min((2*i + ks1), ks2)**2 for i,s in enumerate(dec_strds[:fix_ch_stages])])
    b =  embed_dim * fc_param 
    c =  lower_width **2 * sum([(s**2 if type(s) == int else s[0] * s[1]) * min(2*(fix_ch_stages + i) + ks1, ks2)  **2 for i, s in enumerate(dec_strds[fix_ch_stages:])])
    fc_dim = int(np.roots([a,b,c - decoder_size]).max())

    return fc_dim


def compute_nerv_dim_pe(pe_levels:int, modelsize:int, fc_h:int, fc_w:int, reduce:int, ks1:int, ks2:int, dec_strds:int, lower_width:int):
    embed_param = 0
    embed_dim = pe_levels * 2
    fc_param = fc_h * fc_w
    return compute_nerv_dim(modelsize, reduce, ks1, ks2, dec_strds, lower_width, embed_param, embed_dim, fc_param)

def compute_nerv_dim_enc(enc_strds, enc_dims, modelsize, frame_hw, full_data_length, reduce, ks1, ks2, dec_strds, lower_width):
    total_enc_strds = np.prod(enc_strds)
    embed_hw = np.prod(frame_hw) / total_enc_strds**2
    enc_dim1, embed_ratio = [float(x) for x in enc_dims.split('_')]
    embed_dim = int(embed_ratio * modelsize * 1e6 / full_data_length / embed_hw) if embed_ratio < 1 else int(embed_ratio) 
    embed_param = float(embed_dim) / total_enc_strds**2 * np.prod(frame_hw) * full_data_length
    enc_dim = f'{int(enc_dim1)}_{embed_dim}' 
    fc_param = (np.prod(enc_strds) // np.prod(dec_strds))**2 * 9

    return compute_nerv_dim(modelsize, reduce, ks1, ks2, dec_strds, lower_width, embed_param, embed_dim, fc_param), enc_dim


def get_grid(flow):
    m, n = flow.shape[-2:]
    shifts_x = torch.arange(0, n, 1, dtype=torch.float32, device=flow.device)
    shifts_y = torch.arange(0, m, 1, dtype=torch.float32, device=flow.device)
    shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x)

    grid_dst = torch.stack((shifts_x, shifts_y)).unsqueeze(0)
    workspace = torch.tensor([(n - 1) / 2, (m - 1) / 2]).view(1, 2, 1, 1).to(flow.device)

    flow_grid = ((flow + grid_dst) / workspace - 1).permute(0, 2, 3, 1)

    return flow_grid


def resample(feats, flow):
    scale_factor = (float(feats.shape[-2]) / flow.shape[-2], float(feats.shape[-1]) / flow.shape[-1])
    flow = torch.nn.functional.interpolate(
        flow, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    flow = flow * max(scale_factor)
    flow_grid = get_grid(flow)
    warped_feats = F.grid_sample(feats, flow_grid, mode="bilinear", padding_mode="border")
    return warped_feats


def get_grid_hinerv(flow):
    #print(f'hinerv flow shape {flow.shape}')
    m, n = flow.shape[-2:]
    shifts_x = torch.arange(0, n, 1, dtype=torch.float32, device=flow.device)
    shifts_y = torch.arange(0, m, 1, dtype=torch.float32, device=flow.device)
    shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x)

    grid_dst = torch.stack((shifts_x, shifts_y)).unsqueeze(0)
    #print(f'hinerv grid_dst shape {grid_dst.shape}')
    workspace = torch.tensor([(n - 1) / 2, (m - 1) / 2]).view(1, 2, 1, 1).to(flow.device)

    flow_grid = ((flow + grid_dst) / workspace - 1).permute(0, 2, 3, 1)
    #print(f'hinerv flow_grid shape {flow_grid.shape}')

    return flow_grid


def assert_shape(x: torch.Tensor, shape: tuple):
    assert tuple(x.shape) == tuple(shape), f'shape: {x.shape}     expected: {shape}'


def compute_paddings(output_patchsize=(1, 120, 120), scales=((1, 5, 5), (1, 4, 4), (1, 3, 3), (1, 2, 2)),
                     kernel_sizes=((0, 1, 1), (0, 1, 1), (0, 1, 1), (0, 1, 1)), 
                     depths=(3, 3, 3, 1), resize_methods='trilinear'):
    """
    Compute the required padding sizes.
    This is just an approximation to the padding required. The exact way is to calculate the boundary coordinates of the patches.
    Length of returned padding == len(scales) + 1 == len(padding_per_layer) + 1 == len(depths) + 1
    """
    assert len(scales) == len(depths) == len(kernel_sizes)    
    paddings = np.zeros(3, dtype=np.int32)
    scales = np.array(scales)
    kernel_sizes = np.array(kernel_sizes).clip(min=1)
    paddings_reversed = []
    for i in reversed(range(len(scales))):
        assert np.all((kernel_sizes[i] - 1) % 2 == 0)
        assert np.all(scales[i] >= 1.), 'scales must be positive'
        assert np.all(output_patchsize % scales[i] == 0), 'output_patchsize must be divisble by scales'
        paddings += depths[i] * (kernel_sizes[i] - 1) // 2
        if resize_methods == 'trilinear':
            paddings_reversed.append(paddings.tolist())
            paddings = np.round(paddings / scales[i]).astype(np.int32) + (scales[i] > 1.)
        elif resize_methods == 'nearest':
            paddings_reversed.append(paddings.tolist())
            paddings = np.round(paddings / scales[i]).astype(np.int32)
        elif resize_methods in ['conv1x1', 'conv3x3']:
            paddings_reversed.append(paddings.tolist())
            paddings = np.ceil(paddings / scales[i]).astype(np.int32) + (scales[i] > 1.)
            if resize_methods == 'conv1x1':
                pass
            elif resize_methods == 'conv3x3':
                paddings += np.array([0, 1, 1])
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    paddings_reversed.append(paddings.tolist())
    return list(reversed(paddings_reversed))


def get_encoding_cfg(enc_type, depth, size, **kwargs):
    assert enc_type in ['base', 'upsample']
    assert (enc_type == 'base' and depth == 0) or (enc_type in ['upsample'] and depth >= 0)

    if kwargs['type'] == 'none':
        return {}

    # Positional
    Bt, Lt, Bs, Ls = kwargs['pe']
    Lt, Ls = int(Lt), int(Ls)

    # Grid
    if len(kwargs['grid_size']) == 4:
        # Learned/Local learned (full)
        T_grid, H_grid, W_grid, C_grid = kwargs['grid_size']
        if 'grid_depth_scale' in kwargs:
            T_grid_scale, H_grid_scale, W_grid_scale, C_grid_scale = kwargs['grid_depth_scale']
        else:
            T_grid_scale = H_grid_scale = W_grid_scale = C_grid_scale = 1.
    elif len(kwargs['grid_size']) == 2:
        # Local learned (temporal only)
        T_grid, C_grid = kwargs['grid_size']
        H_grid, W_grid = 1, 1
        if 'grid_depth_scale' in kwargs:
            T_grid_scale, C_grid_scale = kwargs['grid_depth_scale']
            H_grid_scale, W_grid_scale = 1, 1
        else:
            T_grid_scale = H_grid_scale = W_grid_scale = C_grid_scale = 1.
    else:
        raise NotImplementedError

    T_grid = max(int(T_grid * T_grid_scale ** depth), 1) if T_grid != -1 else size[0]
    H_grid = max(int(H_grid * H_grid_scale ** depth), 1) if H_grid != -1 else size[1]
    W_grid = max(int(W_grid * W_grid_scale ** depth), 1) if W_grid != -1 else size[2]
    C_grid = max(int(C_grid * C_grid_scale ** depth), 1) if C_grid != -1 else size[3]

    if len(kwargs['grid_size']) == 4:
        kwargs['grid_size'] = [T_grid, H_grid, W_grid, C_grid]
    elif len(kwargs['grid_size']) == 2:
        kwargs['grid_size'] = [T_grid, C_grid]
    else:
        raise NotImplementedError

    # All configs
    enc_cfg = {
        'type': kwargs['type'],
        # frequency encoding
        'Bt': Bt, 'Lt': Lt, 'Bs': Bs, 'Ls': Ls, 'no_t': kwargs['pe_no_t'],
        # grid encoding
        'grid_size': kwargs['grid_size'], 'grid_level': kwargs['grid_level'], 'grid_level_scale': kwargs['grid_level_scale'],
        'grid_init_scale': kwargs['grid_init_scale'], 'align_corners': kwargs['align_corners']
    }

    return enc_cfg


def unwrap_model(model):
    model = model._orig_mod if hasattr(model, '_orig_mod') else model # For compiled models
    model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model # For DDP models
    return model
