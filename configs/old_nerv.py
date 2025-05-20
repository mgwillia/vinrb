def get_config(data_shape, patch_size=None, num_frames=132):
    config = {
        'pe_lbase': 1.25, 
        'pe_levels': 40,
        'stem_dim_num': '512_1', # NOTE: change for size
        'fc_dim': 24, # NOTE: change for size
        'lower_width': 96, # NOTE: change for size
        'decoder_strides': [5, 3, 2, 2, 2],
        'reduce': 2,
        'expansion': [1, 1, 1, 1, 1],
        'fc_hw': '9_16',
        'out_bias': 'sigmoid',
        'stem_type': 'nerv',
        'use_norm': False,
        'use_fuse_t': False,
        'use_old_upconv': True,
        'patch_size': patch_size,
        'data_shape': data_shape,
        'group_size': 1,
    }

    return config