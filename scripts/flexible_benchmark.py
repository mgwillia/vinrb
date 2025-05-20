import argparse
import json
import sys
import os
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv('VINRB_PATH'))

from configs import get_base_configs
from trainers import NeRVTrainer
from utils import benchmark, set_pruning, set_quantization, initial_parametrizations
from video_encoders import NeRVEncoder

def main(args):

    # 0. Set up data
    data_path = args.data_path
    save_path = os.path.join(args.save_path, f'{args.model_type}_{args.checkpoint_suffix}')
    os.makedirs(save_path, exist_ok=True)
    num_frames = args.num_frames
    data_shape = [int(x) for x in args.data_shape.split('_')]
    if args.patch_size is not None:
        patch_size = [int(x) for x in args.patch_size.split('_')]
    else:
        patch_size = args.patch_size

    model_type = args.model_type
    checkpoint_suffix = args.checkpoint_suffix
    base_configs = get_base_configs(data_shape, patch_size, num_frames)
    base_config = base_configs[model_type]

    if args.config_override_path is not None:
        with open(args.config_override_path, 'r') as f:
            config_override = json.load(f)
        for key, val in config_override.items():
            base_config[key] = val


    encoder_config = base_config
    positional_encoding_lookup = {
        'old_nerv': 'nerv',
        'nerv': 'nerv',
        'hnerv': 'hnerv',
        'enerv': 'enerv',
        'ffnerv': 'ffnerv',
        'hinerv': 'hinerv',
    }
    positional_encoding = positional_encoding_lookup[model_type] if args.positional_encoding is None else args.positional_encoding

    extra_params = {
        'checkpoint_path': f'checkpoints/{model_type}_{checkpoint_suffix}_{data_path.split("/")[-1]}_config/',
        'checkpoint_freq': args.checkpoint_freq,
        'resume': True
    }
    if 'hinerv' in model_type:
        extra_params['batch_size'] = int(
            (encoder_config['data_shape'][0] * encoder_config['data_shape'][1]) / 
            (encoder_config['nerv_patch_size'][1]*encoder_config['nerv_patch_size'][2])
        )
        extra_params['lr'] = 2e-3 if args.learning_rate is None else args.learning_rate
    else:
        extra_params['lr'] = 1e-4 if args.learning_rate is None else args.learning_rate
    extra_params['warmup_lr'] = extra_params['lr'] / 100
    extra_params['min_lr'] = extra_params['lr'] / 100
    extra_params['warmup_epochs'] = args.warmup_epochs


    # 1. Initialize Encoder
    encoder = NeRVEncoder(
            data_path=data_path,
            positional_encoding=positional_encoding,
            **encoder_config,
    )

    # 2. Train Encoder (measure training time)
    trainer = NeRVTrainer(encoder, 
                            encoder.data_pipeline,
                            num_iters=args.train_epochs, 
                            loss='mse', 
                            save_path=save_path,
                            skip_save=True,
                            extra_params = extra_params,
                            batch_size = extra_params.get('batch_size', 1),
                            num_workers=4
    )
    benchmark(encoder, trainer, out_dir=f'output/{model_type}_{checkpoint_suffix}_{data_path.split("/")[-1]}_metrics')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process benchmark args')
    parser.add_argument(
        '--data-path', type=str, default='examples/bunny_debug', help='path to load video frames',
    )
    parser.add_argument(
        '--save-path', type=str, default='examples/bunny_example', help='path to save decoded video outputs',
    )
    parser.add_argument(
        '--data-shape', type=str, default='720_1280', help='str of format height_width',
    )
    parser.add_argument(
        '--patch-size', type=str, default=None, help='size of individual patches for NIRVANA-style encoding',
    )
    parser.add_argument(
        '--num-frames', type=int, default=132, help='number of frames in video',
    )
    parser.add_argument(
        '--model-type', type=str, default='nerv', help='type of INR used to represent the video',
    )
    parser.add_argument(
        '--checkpoint-suffix', type=str, default='default', help='key identifier for checkpoints for this experiment',
    )
    parser.add_argument(
        '--positional-encoding', type=str, default=None
    )
    parser.add_argument(
        '--config-override-path', type=str, default=None
    )
    parser.add_argument(
        '--checkpoint-freq', type=int, default=5
    )
    parser.add_argument(
        '--learning-rate', type=float, default=None
    )
    parser.add_argument(
        '--train-epochs', type=int, default=300
    )
    parser.add_argument(
        '--prune-epochs', type=int, default=60
    )
    parser.add_argument(
        '--quant-epochs', type=int, default=30
    )
    parser.add_argument(
        '--warmup-epochs', type=int, default=1
    )

    args = parser.parse_args()

    main(args)