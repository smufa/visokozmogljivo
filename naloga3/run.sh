#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --job-name=gray-scott
#SBATCH --gpus=1
#SBATCH --output=gray_scott.log

module load CUDA

make all

# Input data file (system parameters)
DATA_FILE="${1:-data/input.txt}"
IMAGE_FILE="${2}"

# Read simulation parameters from input file
while IFS=' = ' read -r key value; do
  case "$key" in
    delta_t)      DELTA_T="$value" ;;
    Du)           DIFFUSION_RATE_A="$value" ;;
    Dv)           DIFFUSION_RATE_B="$value" ;;
    F)            FEED_RATE="$value" ;;
    k)            KILL_RATE="$value" ;;
    time_steps)   TIME_STEPS="$value" ;;
  esac
done < "$DATA_FILE"


# Create results directories
mkdir -p results/cpu
mkdir -p results/cuda

# Run CPU implementation
echo "Running CPU implementation..."
./cpu.out --diffusion_rate_a $DIFFUSION_RATE_A --diffusion_rate_b $DIFFUSION_RATE_B --feed_rate $FEED_RATE --kill_rate $KILL_RATE --time_steps $TIME_STEPS --delta_t $DELTA_T --image_file "$IMAGE_FILE" > results/cpu/cpu.log

# Run CUDA implementation
echo "Running CUDA implementation..."
./cuda.out --diffusion_rate_a $DIFFUSION_RATE_A --diffusion_rate_b $DIFFUSION_RATE_B --feed_rate $FEED_RATE --kill_rate $KILL_RATE --time_steps $TIME_STEPS --delta_t $DELTA_T --image_file "$IMAGE_FILE" > results/cuda/cuda.log