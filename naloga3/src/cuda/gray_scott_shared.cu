#include "../helpers.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <string>

__device__ float &cudaImageAt(float *image_data, int width, int height, int x,
                              int y, int channel) {
  x = (x % width + width) % width;
  y = (y % height + height) % height;
  size_t index = (y * width + x) * 2 + channel;
  return image_data[index];
}

__global__ void grayScottKernel(float *image, float *image_copy, int width,
                                int height, int channels,
                                double diffusion_rate_a,
                                double diffusion_rate_b, double feed_rate,
                                double kill_rate, // time_steps removed
                                double delta_t) {
  extern __shared__ float s_mem[];

  int tile_dim_x_halo = blockDim.x + 2;
  int tile_dim_y_halo = blockDim.y + 2;
  int shared_mem_per_component = tile_dim_x_halo * tile_dim_y_halo;

  float* s_U = s_mem;
  float* s_V = &s_mem[shared_mem_per_component];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid_in_block = ty * blockDim.x + tx;

  // Load data into shared memory (U and V components)
  for (int i = tid_in_block; i < shared_mem_per_component; i += blockDim.x * blockDim.y) {
    int s_y_current = i / tile_dim_x_halo; // y-index in shared memory tile
    int s_x_current = i % tile_dim_x_halo; // x-index in shared memory tile

    // Global coordinates for this shared memory point
    int gx_load = (blockIdx.x * blockDim.x) + s_x_current - 1;
    int gy_load = (blockIdx.y * blockDim.y) + s_y_current - 1;

    s_U[s_y_current * tile_dim_x_halo + s_x_current] = cudaImageAt(image, width, height, gx_load, gy_load, 0);
    s_V[s_y_current * tile_dim_x_halo + s_x_current] = cudaImageAt(image, width, height, gx_load, gy_load, 1);
  }
  __syncthreads(); // Ensure all shared memory is loaded

  // Global coordinates for the pixel this thread is responsible for
  int gx = blockIdx.x * blockDim.x + tx;
  int gy = blockIdx.y * blockDim.y + ty;

  if (gx < width && gy < height) {
    // Local indices for accessing the center of the 3x3 stencil in shared memory
    int center_sx = tx + 1;
    int center_sy = ty + 1;

    float U_val = s_U[center_sy * tile_dim_x_halo + center_sx];
    float V_val = s_V[center_sy * tile_dim_x_halo + center_sx];

    // Calculate Laplacian using shared memory
    float laplacian_U = s_U[center_sy * tile_dim_x_halo + (center_sx + 1)] +       // E
                        s_U[center_sy * tile_dim_x_halo + (center_sx - 1)] +       // W
                        s_U[(center_sy + 1) * tile_dim_x_halo + center_sx] +       // S
                        s_U[(center_sy - 1) * tile_dim_x_halo + center_sx] -       // N
                        4.0f * U_val;

    float laplacian_V = s_V[center_sy * tile_dim_x_halo + (center_sx + 1)] +       // E
                        s_V[center_sy * tile_dim_x_halo + (center_sx - 1)] +       // W
                        s_V[(center_sy + 1) * tile_dim_x_halo + center_sx] +       // S
                        s_V[(center_sy - 1) * tile_dim_x_halo + center_sx] -       // N
                        4.0f * V_val;

    float UV2 = U_val * V_val * V_val;

    cudaImageAt(image_copy, width, height, gx, gy, 0) =
        U_val + (float)delta_t *
                ((float)diffusion_rate_a * laplacian_U - UV2 + (float)feed_rate * (1.0f - U_val));
    cudaImageAt(image_copy, width, height, gx, gy, 1) =
        V_val + (float)delta_t * ((float)diffusion_rate_b * laplacian_V + UV2 -
                       ((float)feed_rate + (float)kill_rate) * V_val);
    __syncthreads();
    
    // Swap buffers (still using global memory for this part)
    cudaImageAt(image, width, height, gx, gy, 0) =
        cudaImageAt(image_copy, width, height, gx, gy, 0);
    cudaImageAt(image, width, height, gx, gy, 1) =
        cudaImageAt(image_copy, width, height, gx, gy, 1);
    __syncthreads();
  }
}

int main(int argc, char *argv[]) {
  std::cout << "Gray-Scott CUDA implementation" << std::endl;

  double diffusion_rate_a = 0.0;
  double diffusion_rate_b = 0.0;
  double feed_rate = 0.0;
  double kill_rate = 0.0;
  int time_steps = 0;
  double delta_t = 0.0;
  std::string image_file = "";
  std::string output_file = "output_cuda_shared.png";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--diffusion_rate_a") {
      diffusion_rate_a = std::stod(argv[++i]);
    } else if (arg == "--diffusion_rate_b") {
      diffusion_rate_b = std::stod(argv[++i]);
    } else if (arg == "--feed_rate") {
      feed_rate = std::stod(argv[++i]);
    } else if (arg == "--kill_rate") {
      kill_rate = std::stod(argv[++i]);
    } else if (arg == "--time_steps") {
      time_steps = std::stoi(argv[++i]);
    } else if (arg == "--delta_t") {
      delta_t = std::stod(argv[++i]);
    } else if (arg == "--image_file") {
      image_file = argv[++i];
    } else if (arg == "--output_file") {
      output_file = argv[++i];
    }
  }

  std::cout << "diffusion_rate_a: " << diffusion_rate_a << std::endl;
  std::cout << "diffusion_rate_b: " << diffusion_rate_b << std::endl;
  std::cout << "feed_rate: " << feed_rate << std::endl;
  std::cout << "kill_rate: " << kill_rate << std::endl;
  std::cout << "time_steps: " << time_steps << std::endl;
  std::cout << "delta_t: " << delta_t << std::endl;
  std::cout << "image_file: " << image_file << std::endl;

  if (diffusion_rate_a == 0.0 || diffusion_rate_b == 0.0 || feed_rate == 0.0 ||
      kill_rate == 0.0 || delta_t == 0.0) {
    std::cerr << "Error: diffusion_rate_a, diffusion_rate_b, feed_rate, "
                 "kill_rate, and delta_t must be set."
              << std::endl;
    return 1;
  }

  if (!image_file.empty()) {
    int width, height, channels;
    float *img = loadImage(image_file.c_str(), &width, &height, &channels);
    if (img == nullptr) {
      std::cerr << "Error loading image: " << stbi_failure_reason()
                << std::endl;
    } else {
      std::cout << "Loaded image: " << image_file << " (" << width << "x"
                << height << ", " << channels << " channels)" << std::endl;

      // Allocate device memory
      float *d_image, *d_image_copy;
      size_t image_size = width * height * channels;
      cudaMallocManaged(&d_image, image_size * sizeof(float));
      cudaMallocManaged(&d_image_copy, image_size * sizeof(float));

      // Create CUDA events for timing
      cudaEvent_t start_event, stop_event;
      cudaEventCreate(&start_event);
      cudaEventCreate(&stop_event);
      float milliseconds = 0;

      // Time memory copy to device
      cudaEventRecord(start_event);
      cudaMemcpy(d_image, img, image_size * sizeof(float),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(d_image_copy, img, image_size * sizeof(float),
                 cudaMemcpyHostToDevice);
      cudaEventRecord(stop_event);
      cudaEventSynchronize(stop_event);
      cudaEventElapsedTime(&milliseconds, start_event, stop_event);
      std::cout << "Memory copy to device time: " << milliseconds << " ms" << std::endl;

      // Define kernel
      dim3 block_size(16, 16);
      dim3 grid_size((width + block_size.x - 1) / block_size.x,
                     (height + block_size.y - 1) / block_size.y);
      
      size_t shared_mem_size = 2 * (block_size.x + 2) * (block_size.y + 2) * sizeof(float);

      // Time kernel execution
      cudaEventRecord(start_event);
      for (int step = 0; step < time_steps; ++step) {
        grayScottKernel<<<grid_size, block_size, shared_mem_size>>>(
            d_image, d_image_copy, width, height, channels, diffusion_rate_a,
            diffusion_rate_b, feed_rate, kill_rate, delta_t);
        // cudaDeviceSynchronize(); // Not needed if timing the whole loop
      }
      cudaEventRecord(stop_event);
      cudaEventSynchronize(stop_event); // Ensure all kernel launches are complete
      cudaEventElapsedTime(&milliseconds, start_event, stop_event);
      std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

      // Time memory copy back to host
      cudaEventRecord(start_event);
      cudaMemcpy(img, d_image, image_size * sizeof(float),
                 cudaMemcpyDeviceToHost);
      cudaEventRecord(stop_event);
      cudaEventSynchronize(stop_event);
      cudaEventElapsedTime(&milliseconds, start_event, stop_event);
      std::cout << "Memory copy to host time: " << milliseconds << " ms" << std::endl;

      // Write image
      writeImage(output_file.c_str(), width, height, channels, img); // Changed output filename

      // Free memory
      cudaEventDestroy(start_event);
      cudaEventDestroy(stop_event);
      cudaFree(d_image);
      cudaFree(d_image_copy);

      delete[] img;
    }
  }
  return 0;
}