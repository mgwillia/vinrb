# Video Implicit Neural Representation Benchmark (VINRB)
Official code implementation for "How to Design and Train Your Implicit Neural Representation for Video Compression." 
We currently support NeRV, E-NeRV, HNeRV, HiNeRV, FFNeRV, DiffNeRV, DiVNeRV, with support for most meaningful ablations and combinations of their many components.
This repository serves both as a benchmark of progress so far and a framework that enables future work on new video compression INRs.

## Environment Setup

### pip install
For best results, use Python 3.9.6 and cuda 11.8.0.
To match our environment compile on NVIDIA RTXA5000. 
The code runs on other cards as well. 
We test and verify on RTXA4000 and certain HiNeRV settings require a card with a lot of GPU RAM (we use an RTXA6000).

```
pip install wheel
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### dotenv
You need 2 variables in a .env at the root of the project, `VENV_PATH` and `VINRB_PATH`. 

`VENV_PATH` is the absolute path to your python virtual environment. There should be no trailing slash. 
`VINRB_PATH` is the absolute path to the root of the project. There should be no trailing slash.

Example `.env` file contents:
```
VENV_PATH=/your_path/vinrb_env
VINRB_PATH=/your_path/vinrb
```

## Sample Command

To run our RNeRV configuration, in "short" mode (2 minutes on our RTXA5000), run

```
python scripts/flexible_benchmark.py --data-shape 1080_1920 --num-frames 600 --data-path /path/to/UVG/honeybee_1080 --save-path results/honeybee_1080 --model-type enerv --checkpoint-suffix rnerv-short-1_5 --config-override-path configs/overrides/rnerv-1_5.json --positional-encoding ffnerv --train-epochs 11 --warmup-epochs 0 
```

Compressed INRs are saved to `results`, model weights are saved to `checkpoints`, csvs with training progress (quality, wall time) are saved in `output`, and PSNR/bpp are printed to standard out once training completes.