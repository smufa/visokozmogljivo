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
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    // Loop for time_steps removed from here
    // Access pixel data here
    float U = cudaImageAt(image, width, height, x, y, 0);
    float V = cudaImageAt(image, width, height, x, y, 1);

    // Calculate Laplacian
    float laplacian_U = cudaImageAt(image, width, height, x + 1, y, 0) +
                        cudaImageAt(image, width, height, x - 1, y, 0) +
                        cudaImageAt(image, width, height, x, y + 1, 0) +
                        cudaImageAt(image, width, height, x, y - 1, 0) -
                        4.0f * U;

    float laplacian_V = cudaImageAt(image, width, height, x + 1, y, 1) +
                        cudaImageAt(image, width, height, x - 1, y, 1) +
                        cudaImageAt(image, width, height, x, y + 1, 1) +
                        cudaImageAt(image, width, height, x, y - 1, 1) -
                        4.0f * V;

    float UV2 = U * V * V;

    cudaImageAt(image_copy, width, height, x, y, 0) =
        U + (float)delta_t *
                ((float)diffusion_rate_a * laplacian_U - UV2 + (float)feed_rate * (1 - U));
    cudaImageAt(image_copy, width, height, x, y, 1) =
        V + (float)delta_t * ((float)diffusion_rate_b * laplacian_V + UV2 -
                       ((float)feed_rate + (float)kill_rate) * V);
    __syncthreads();
    // Swap buffers
    
    cudaImageAt(image, width, height, x, y, 0) =
        cudaImageAt(image_copy, width, height, x, y, 0);
    cudaImageAt(image, width, height, x, y, 1) =
        cudaImageAt(image_copy, width, height, x, y, 1);
    __syncthreads();
    // Loop for time_steps removed from here
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
std::string output_file = "output_cuda.png";

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

      // Time kernel execution
      cudaEventRecord(start_event);
      for (int step = 0; step < time_steps; ++step) {
        grayScottKernel<<<grid_size, block_size>>>(
            d_image, d_image_copy, width, height, channels, diffusion_rate_a,
            diffusion_rate_b, feed_rate, kill_rate, delta_t); // time_steps removed from call
        // cudaDeviceSynchronize(); // Synchronize after each step // Not needed if timing the whole loop
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
      writeImage(output_file.c_str(), width, height, channels, img);

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