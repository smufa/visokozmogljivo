#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "Image.hpp"

#define COLOR_CHANNELS 0

struct gpuImage {
  int width;
  int height;
  int channels;
  float *data;
};

__device__ float at(gpuImage img, int x, int y, int c) {
    if (x < 0 || x >= img.width || y < 0 || y >= img.height || c < 0 || c >= img.channels) {
        static float dummy = 0.0f;
        return dummy; 
    }
    
    int index = (y * img.width + x) * img.channels + c;
    
    return img.data[index];
}

__device__ void set(gpuImage img, int x, int y, int c, float value) {
  if (x < 0 || x >= img.width || y < 0 || y >= img.height || c < 0 || c >= img.channels) {
      return; 
  }
  
  int index = (y * img.width + x) * img.channels + c;
  img.data[index] = value;
}


__global__ void rgb_to_yuv(const gpuImage in_img, gpuImage out_img) {
    set(out_img, 1, 1, 0, at(in_img, 0, 0, 0));
}

gpuImage allocateImageOnGPU(const Image& img) {
    gpuImage out_img;
    out_img.width = img.getWidth();
    out_img.height = img.getHeight();
    out_img.channels = img.getChannels();
    cudaError_t err = cudaMalloc(&out_img.data, img.getDataSize() * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate CUDA memory: " + 
                                std::string(cudaGetErrorString(err)));
    }

    // Copy data from host to device
    const std::vector<float>& hostData = img.getData();
    err = cudaMemcpy(out_img.data, hostData.data(), img.getDataSize() * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        // Free allocated memory before throwing
        cudaFree(out_img.data);
        throw std::runtime_error("Failed to copy data to CUDA memory: " + 
                                std::string(cudaGetErrorString(err)));
    }
    
    return out_img;
}

Image retrieveImageFromGPU(const gpuImage& gpu_img) {
    // Create a new host image with the same dimensions
    Image host_img(gpu_img.width, gpu_img.height, gpu_img.channels);
    
    // Get a reference to the host data vector for direct access
    const std::vector<float>& hostData = host_img.getData();
    
    // Calculate the size of data to copy
    size_t dataSize = gpu_img.width * gpu_img.height * gpu_img.channels * sizeof(float);
    
    // Copy data from device to host
    cudaError_t err = cudaMemcpy(const_cast<float*>(hostData.data()), 
                                gpu_img.data, 
                                dataSize, 
                                cudaMemcpyDeviceToHost);
    
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy data from CUDA memory: " + 
                                std::string(cudaGetErrorString(err)));
    }
    
    return host_img;
}

int main(int argc, char *argv[]) {

  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <input> <output>\n";
    return EXIT_FAILURE;
  }

  // Process parameters
  std::string input_filename = argv[1];
  std::string output_filename = argv[2];

  try {
    // Load image
    Image input = Image::load(input_filename);

    // Process image
    // Setup Thread organization
    dim3 blockSize(16, 16);
    dim3 gridSize((input.getHeight() - 1) / blockSize.x + 1,
                  (input.getWidth() - 1) / blockSize.y + 1);
    // dim3 gridSize(1, 1);

    
    // Use CUDA events to measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Copy image to device and run kernel
    cudaEventRecord(start);
    gpuImage imageIn = allocateImageOnGPU(input);
    gpuImage imageOut = allocateImageOnGPU(input);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    // Print time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel Execution time is: %0.3f milliseconds \n", milliseconds);

    Image output = retrieveImageFromGPU(imageOut);
    output.save(output_filename);
    cudaFree(imageOut.data);

  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
