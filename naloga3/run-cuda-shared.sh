#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --job-name=gray-scott-cuda-shared
#SBATCH --gpus=1

module load CUDA

make cuda_shared

# Input data file (system parameters)
DATA_FILE="data/input.json"

# Read simulation parameters from input file
TIME_STEPS=$(jq -r .time_steps "$DATA_FILE")
DELTA_T=$(jq -r .delta_t "$DATA_FILE")
DIFFUSION_RATE_A=$(jq -r .Du "$DATA_FILE")
DIFFUSION_RATE_B=$(jq -r .Dv "$DATA_FILE")
FEED_RATE=$(jq -r .F "$DATA_FILE")
KILL_RATE=$(jq -r .k "$DATA_FILE")

# Create results directory if it doesn't exist
mkdir -p results

# Iterate over all PNG files in the data directory
for IMAGE_FILE in data/*.png; do
  # Extract image file name without extension
  IMAGE_NAME=$(basename "$IMAGE_FILE" .png)

  # Run CUDA shared memory implementation and redirect output to file
  echo "Running CUDA shared memory implementation for $IMAGE_FILE..."
  ./cuda_shared.out --diffusion_rate_a $DIFFUSION_RATE_A --diffusion_rate_b $DIFFUSION_RATE_B --feed_rate $FEED_RATE --kill_rate $KILL_RATE --time_steps $TIME_STEPS --delta_t $DELTA_T --image_file "$IMAGE_FILE"  --output_file "results/shared${IMAGE_NAME}.png" &> "results/${IMAGE_NAME}_shared.log"
done