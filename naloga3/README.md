# Gray-Scott Model Implementation

This project implements the Gray-Scott model of a reaction-diffusion system. It includes separate implementations for CPU (C++) and GPU (CUDA).

## Project Structure

The project is organized as follows:

```
naloga3/
├── src/
│   ├── cpu/
│   │   └── gray_scott_cpu.cpp: Main source file for the CPU implementation.
│   └── cuda/
│       └── gray_scott_cuda.cu: Main source file for the CUDA implementation.
├── results/
│   ├── cpu/: Output directory for results from the CPU implementation.
│   └── cuda/: Output directory for results from the CUDA implementation.
├── data/
│   └── input.txt: Input data file containing system parameters.
├── run.sh: SLURM run script for executing the simulations.
└── Makefile: Build file for compiling the CPU and CUDA implementations.
```

## Implementation Details

*   **CPU (C++)**: The `src/cpu/gray_scott_cpu.cpp` file contains the main function for the CPU implementation.
*   **CUDA**: The `src/cuda/gray_scott_cuda.cu` file contains the main function for the CUDA implementation.

## Building the Project

The `Makefile` provides targets for building both the CPU and CUDA implementations:

*   `make cpu`: Compiles the C++ implementation.
*   `make cuda`: Compiles the CUDA implementation.
*   `make all`: Compiles both implementations.
*   `make clean`: Removes compiled files.

## Running the Simulation


The `run.sh` script is a SLURM script that compiles and runs both the CPU and CUDA implementations.
It accepts the following optional arguments:

*   Input data file path (defaults to `data/input.txt`)
*   Starting image file path

To run the simulation, submit the `run.sh` script to the SLURM queue:
```bash
sbatch run.sh [input_file] [image_file]
```
For example:
```bash
sbatch run.sh data/input.txt initial_image.png
```
## Input Data

The `data/input.txt` file contains the system parameters used by both implementations. The parameters are defined as follows:

```
delta_t = 1
Du = 0.16
Dv = 0.08
F = 0.060
k = 0.062
```

## Results

The simulation results are saved in the `results/` directory, with separate subdirectories for the CPU and CUDA implementations.