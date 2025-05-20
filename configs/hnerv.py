def get_config(data_shape, patch_size=None, num_frames=132):
    config = {
        'encoder_strides': [5, 3, 2, 2, 2],
        'decoder_strides': [5, 3, 2, 2, 2],
        'encoder_dims': '64_16',
        'target_modelsize': 1.5,
        'frame_hw': (1080, 1920),
        'full_data_length': num_frames,
        'ks': '1_5',
        'reduce': 1.2,
        'patch_size': patch_size,
        'data_shape': data_shape,
        'group_size': 1,
        'is_hnerv': True,
    }

    return config