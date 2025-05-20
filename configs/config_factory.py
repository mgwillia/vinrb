from .enerv import get_config as get_enerv_config
from .ffnerv import get_config as get_ffnerv_config
from .hnerv import get_config as get_hnerv_config
from .hinerv import get_config as get_hinerv_config
from .old_nerv import get_config as get_oldnerv_config
from .nerv import get_config as get_nerv_config


def get_base_configs(data_shape, patch_size=None, num_frames=132):
    configs = {}
    configs['old_nerv'] = get_oldnerv_config(data_shape, patch_size, num_frames)
    configs['nerv'] = get_nerv_config(data_shape, patch_size, num_frames)
    configs['enerv'] = get_enerv_config(data_shape, patch_size, num_frames)
    configs['hnerv'] = get_hnerv_config(data_shape, patch_size, num_frames)
    configs['ffnerv'] = get_ffnerv_config(data_shape, patch_size, num_frames)
    configs['hinerv'] = get_hinerv_config(data_shape, patch_size, num_frames)
    configs['hinerv-80'] = get_hinerv_config(data_shape, patch_size, num_frames, base_grid_first=80)
    configs['hinerv-120'] = get_hinerv_config(data_shape, patch_size, num_frames, base_grid_first=120)
    configs['hinerv-160'] = get_hinerv_config(data_shape, patch_size, num_frames, base_grid_first=160)

    return configs
