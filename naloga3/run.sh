#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --job-name=gray-scott
#SBATCH --gpus=1
#SBATCH --output=gray_scott.log

module load CUDA

make all

# Input data file (system parameters)
DATA_FILE="${1:-data/input.json}"
IMAGE_FILE="${2:-data/square.png}"
OUTPUT_FILE="${3:-results/output.png}"

# Read simulation parameters from input file
TIME_STEPS=$(jq -r .time_steps "$DATA_FILE")
DELTA_T=$(jq -r .delta_t "$DATA_FILE")
DIFFUSION_RATE_A=$(jq -r .Du "$DATA_FILE")
DIFFUSION_RATE_B=$(jq -r .Dv "$DATA_FILE")
FEED_RATE=$(jq -r .F "$DATA_FILE")
KILL_RATE=$(jq -r .k "$DATA_FILE")


# Create results directories
mkdir -p results/cpu
mkdir -p results/cuda

# Run CPU implementation
echo "Running CPU implementation..."
./cpu.out --diffusion_rate_a $DIFFUSION_RATE_A --diffusion_rate_b $DIFFUSION_RATE_B --feed_rate $FEED_RATE --kill_rate $KILL_RATE --time_steps "${TIME_STEPS}" --delta_t $DELTA_T --image_file "$IMAGE_FILE" --output_file "$OUTPUT_FILE"

# Run CUDA implementation
echo "Running CUDA implementation..."
./cuda.out --diffusion_rate_a $DIFFUSION_RATE_A --diffusion_rate_b $DIFFUSION_RATE_B --feed_rate $FEED_RATE --kill_rate $KILL_RATE --time_steps "${TIME_STEPS}" --delta_t $DELTA_T --image_file "$IMAGE_FILE"  --output_file "$OUTPUT_FILE"