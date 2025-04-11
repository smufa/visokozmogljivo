#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --job-name=hpc-assignemnt
#SBATCH --gpus=1
#SBATCH --output=sample_out.log

module load CUDA
nvcc  -diag-suppress 550 -O2 -lm sample.cu -o sample
srun  sample valve.png