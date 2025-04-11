#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --job-name=hpc-assignemnt
#SBATCH --gpus=1
#SBATCH --output=sample_out.log

module load CUDA
nvcc  -diag-suppress 550 -O2 -lm sample.cu -o sample
srun  sample 3 data/720x480.png out/720x480.png  data/1024x768.png out/1024x768.png  data/1920x1200.png out/1920x1200.png  data/3840x2160.png out/3840x2160.png  data/7680x4320.png out/7680x4320.png 