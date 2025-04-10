#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=code_sample
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=sample_out.log

export OMP_PLACES=cores
export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

g++ -O2 -lm --openmp main.cpp -o carve

for file in 720x480.png 1024x768.png 1920x1200.png 3840x2160.png 7680x4320.png
do
  echo $file
  srun ./carve test_images/$file results/$file 128
done
