def get_config(data_shape, patch_size=None, num_frames=132):
    config = {
        'pe_lbase': 1.25, 
        'pe_levels': 80,
        'target_modelsize': 3.0,
        'decoder_strides': [5, 3, 2, 2, 2],
        'fc_hw': '9_16', 
        'stem_type': 'hnerv',
        'patch_size': patch_size,
        'data_shape': data_shape,
        'group_size': 1,
    }

    return config