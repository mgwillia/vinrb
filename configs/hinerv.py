import math


def get_config(data_shape, patch_size=None, num_frames=132, base_grid_first=150):
    local_encoding_config = {
        'type': 'normalized+temp_local_grid',
        'align_corners': False,
        'pe': [1.2, 60, 1.2, 60],
        'pe_no_t': False,
        'grid_size': [-1, 4],
        'grid_level': 3,
        'grid_level_scale': [2.0, 0.5],
        'grid_init_scale': 1e-3,
        'grid_depth_scale': [1.0, 0.5]
    }

    nerv_patch_size = [1, 120, 120]
    scales_t = [1, 1, 1, 1]
    scales_hw = [5, 3, 2, 2]
    base_size = [-1, -1, -1]
    base_grid_size = [base_grid_first, -1, -1, 2] # NOTE: change first value for different size
    base_grid_level = 2
    base_grid_level_scale = [2., 1., 1., 0.5]
    base_grid_init_scale = 1e-3
    base_size = (num_frames // int(math.prod(scales_t)) if base_size[0] == -1 else base_size[0],
                        data_shape[0] // int(math.prod(scales_hw)) if base_size[1] == -1 else base_size[1],
                        data_shape[1] // int(math.prod(scales_hw)) if base_size[2] == -1 else base_size[2])
    base_channels = sum([int(base_grid_size[3] // base_grid_level_scale[3] ** i) for i in range(base_grid_level)])
    base_grid_size[0] = base_grid_size[0] if base_grid_size[0] != -1 else base_size[0]
    base_grid_size[1] = base_grid_size[1] if base_grid_size[1] != -1 else base_size[1]
    base_grid_size[2] = base_grid_size[2] if base_grid_size[2] != -1 else base_size[2]
    base_grid_size[3] = base_grid_size[3] if base_grid_size[3] != -1 else 8

    config = {
        'stem_type': 'hinerv',
        'decoder_strides': scales_hw, 
        'fc_hw': '9_16', 
        'fc_dim': 252, # NOTE: change this value for different size
        'use_linear_head': True, 
        'use_hinerv_upsamplers': True,
        'decoding_blocks': [3, 3, 3, 1],
        'expansion': [4., 4., 4., 1.],
        'nerv_patch_size': nerv_patch_size, 
        'cached': 'patch',
        'stem_ks': 3, 
        'stem_paddings': [-1, -1, -1],
        'paddings':  [-1, -1, -1],
        'zero_bias_init': True,
        'local_encoding_config': local_encoding_config,
        'reduce': 2.0,
        'base_size': base_size,
        'base_channels': base_channels,
        'base_grid_size': base_grid_size,
        'base_grid_level': base_grid_level,
        'base_grid_level_scale': base_grid_level_scale,
        'base_grid_init_scale': base_grid_init_scale,
        'patch_size': patch_size,
        'data_shape': data_shape,
        'group_size': 1,
        'is_hinerv': True,
    }

    return config