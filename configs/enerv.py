def get_config(data_shape, patch_size=None, num_frames=132):
    config = {
        'stem_type': 'enerv',
        'pe_lbase': 1.25, 
        'pe_levels': 80,
        'xy_lbase': 1.25, 
        'xy_levels': 80,
        'manip_lbase': 1.25, 
        'manip_levels': 80,
        'stem_dim_num': '512', # NOTE: change for size
        'fc_dim': 62, # NOTE: change for size
        'lower_width': 28, # NOTE: change for size
        'decoder_strides': [5, 3, 2, 2, 2],
        'fc_hw': '9_16',
        'reduce': 2, 
        'expansion': [3, 3, 3, 3, 3],
        'block_dim': 256, # NOTE: change for size
        'enerv_mlp_dim': 128, # NOTE: change for size
        'out_bias': 'sigmoid',
        'use_norm': True,
        'use_fuse_t': True,
        'use_old_upconv': True,
        'patch_size': patch_size,
        'data_shape': data_shape,
        'group_size': 1,
    }

    return config