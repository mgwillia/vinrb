def get_config(data_shape, patch_size=None, num_frames=132):
    config = {
        'stem_type': 'ffnerv',
        't_dim': [256, 512], 
        'fc_dim': 22, # NOTE: change this for size
        'expansion': [4, 1, 1, 1, 1], # NOTE: change this for size, original uses 8, 1, 1, 1, 1
        'reduce': 2,
        'decoder_strides': [5, 3, 2, 2, 2],
        'fc_hw': '9_16',
        'flow_agg_idxs': [-2, -1, 1, 2],
        'flow_block_offset': 3,
        'wbit': 32,
        'out_bias': 'sigmoid',
        'patch_size': patch_size,
        'data_shape': data_shape,
        'group_size': 1,
        'use_compact_blocks': True,
        'is_ffnerv': True,
        'agg_ind': [-2, -1, 1, 2],
        'loss_weight': 0.1,
        'lower_width': 12, # NOTE: change this for size
    }

    return config