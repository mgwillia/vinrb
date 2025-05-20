import os
import csv

def benchmark(encoder, trainer, out_dir=None, prefix=None):
    num_pixels = encoder.data_pipeline.data_set.num_groups * \
                encoder.data_pipeline.data_set.input_data_shape[1] * \
                encoder.data_pipeline.data_set.input_data_shape[2]

    model_params = sum(p.numel() for p in encoder.net.parameters())
    print(f'Model has {model_params} params without position encoder')
    model_params += sum(p.numel() for p in encoder.positional_encoder.parameters())
    print(f'Model has {model_params} params', flush=True)

    #breakpoint()

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    if prefix is not None:
        print(f'Before {prefix} training, or during if restarting')
        _, compressed_bits = trainer.compress(compression_config={'quant_level': 8})
        metrics = trainer.decode(compression_config={'quant_level': 8})
        print(f'{metrics["psnr"]} PSNR, {metrics["msssim"]} MS-SSIM @ {compressed_bits / num_pixels} bpp')

        if prefix == 'quant':
            _, compressed_bits = trainer.compress(compression_config={'quant_level': 6})
            metrics = trainer.decode(compression_config={'quant_level': 6})
            print(f'{metrics["psnr"]} PSNR, {metrics["msssim"]} MS-SSIM @ {compressed_bits / num_pixels} bpp')
    
    metrics_out = os.path.join(out_dir, f"{encoder.positional_encoding}.csv")
    time_to_train, train_metrics = trainer.train(metrics_out=metrics_out, prefix=prefix)

    for quant_level in [32, 8, 7, 6, 5, 4]:
        if quant_level != 32:
            compression_config = {
                'quant_level': quant_level
            }
        else:
            compression_config = None
        # 3. Encode Video (measure encoding time, bits per pixel)
        print(f'preparing to compress', flush=True)
        time_to_encode, compressed_bits = trainer.compress(compression_config=compression_config)

        print('finished compressing', flush=True)

        # 4. Decode Video (measure decoding time, quality)
        metrics = trainer.decode(compression_config=compression_config)

        print('successfully returned decode', flush=True)

        # 5. Report encoding FPS (accounting for Train + Encode), compressed size, decoding FPS, PSNR/MS-SSIM
        print(f'training time: {time_to_train}, encoding time: {time_to_encode}, total compression time: {time_to_train + time_to_encode}', flush=True)
        print(f'Encoding FPS: {encoder.data_pipeline.data_set.num_groups / (time_to_encode + time_to_train)}', flush=True)
        print(f'bpp: {compressed_bits / num_pixels}', flush=True)
        print(f'Decoding FPS: {encoder.data_pipeline.data_set.num_groups / metrics["decoding_time"]}', flush=True)
        print(f'PSNR: {metrics["psnr"]}, MS-SSIM: {metrics["msssim"]}', flush=True)

def dump_frames(trainer, out_dir=None):
    trainer.skip_save = False
    trainer.save_path = out_dir if out_dir is not None else trainer.save_path
    trainer.decode()
