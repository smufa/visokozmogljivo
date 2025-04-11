#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Include stb_image headers
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define WORKGROUP_SIZE 16
#define BINS 256
#define ARGS_LEN 2

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


__global__ void rgb_to_yuv(unsigned char *input_image,
                           unsigned char *output_image, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int offset = (y * width + x) * 3;

    unsigned char R = input_image[offset];
    unsigned char G = input_image[offset + 1];
    unsigned char B = input_image[offset + 2];

    unsigned char Y = (unsigned char)(0.299f * R + 0.587f * G + 0.114f * B);
    unsigned char U =
        (unsigned char)(-0.168736f * R - 0.331264f * G + 0.5f * B + 128);
    unsigned char V =
        (unsigned char)(0.5f * R - 0.418688f * G - 0.081312f * B + 128);

    output_image[offset] = Y;
    output_image[offset + 1] = U;
    output_image[offset + 2] = V;
  }
}

__global__ void yuv_to_rgb(unsigned char *input_image,
                           unsigned char *output_image, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int offset = (y * width + x) * 3;

    unsigned char Y = input_image[offset];
    unsigned char U = input_image[offset + 1];
    unsigned char V = input_image[offset + 2];

    float U_adj = U - 128.0f;
    float V_adj = V - 128.0f;

    int R = Y + 1.402f * V_adj;
    int G = Y - 0.344136f * U_adj - 0.714136f * V_adj;
    int B = Y + 1.772f * U_adj;

    R = max(0, min(255, R));
    G = max(0, min(255, G));
    B = max(0, min(255, B));

    output_image[offset] = (unsigned char)R;
    output_image[offset + 1] = (unsigned char)G;
    output_image[offset + 2] = (unsigned char)B;
  }
}

__global__ void histogramKernel(
    const unsigned char* image,
    unsigned int* histogram,
    int width,
    int height
) {
    // Create shared memory for the block-local histogram
    __shared__ unsigned int sharedHistogram[256];
    
    // Calculate thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int blockSize = blockDim.x * blockDim.y;
    int threadId = ty * blockDim.x + tx;
    
    // Calculate global pixel coordinates
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    
    // Initialize shared memory histogram bins to zero
    // Each thread initializes one or more bins
    for (int i = threadId; i < 256; i += blockSize) {
        sharedHistogram[i] = 0;
    }
    
    // Synchronize to ensure shared memory is fully initialized
    __syncthreads();
    
    // Process pixel if within image bounds
    if (x < width && y < height) {
        // Calculate pixel index in the image array
        // For an RGB image with interleaved storage:
        // R,G,B,R,G,B,...
        int pixelIndex = (y * width + x) * 3;
        
        // Get the value of the specified channel for this pixel
        unsigned char channelValue = image[pixelIndex];
        
        // Atomically increment the corresponding bin in shared histogram
        atomicAdd(&sharedHistogram[channelValue], 1);
    }
    
    // Wait for all threads in the block to finish processing
    __syncthreads();
    
    // Merge shared histogram into global histogram
    // Each thread updates one or more bins
    for (int i = threadId; i < 256; i += blockSize) {
        if (sharedHistogram[i] > 0) {
            atomicAdd(&histogram[i], sharedHistogram[i]);
        }
    }
}

__global__ void computeCumulativeHistogram(unsigned int *histogram,
                                           unsigned int *minimum) {
  for (int i = 1; i < 256; i++) {
    if (*minimum == 0 && histogram[i] != 0)
      *minimum = histogram[i];
    histogram[i] = histogram[i - 1] + histogram[i];
  }
}

__global__ void histogramEqualizationKernel(unsigned char *input,
                                            unsigned int *cdf, int width,
                                            int height, unsigned int *minimum) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int idx = (y * width + x) * 3;
    unsigned char pixel = input[idx];

    float numerator = cdf[pixel] - *minimum;
    float denominator = width * height - *minimum;
    float result = (numerator / denominator) * 255.0f;

    input[idx] = (unsigned char)max(0, min(255, (int)roundf(result)));
  }
}

int main(int argc, char **argv) {
  if (argc < 4) {
      printf("Usage: %s <iterations> <input_image1> <output_image1> [<input_image2> <output_image2> ...]\n", argv[0]);
      printf("Invalid arguments!\n");
      return -1;
  }

  // Parse the number of iterations
  int iterations = atoi(argv[1]);
  if (iterations <= 0) {
      printf("Number of iterations must be a positive integer\n");
      return -1;
  }

  // Check if we have pairs of input/output files
  if ((argc - 2) % 2 != 0) {
      printf("Each input image must have a corresponding output image\n");
      return -1;
  }

  int numFiles = (argc - 2) / 2;
  printf("-------------- Starting GPU compute (CUDA) --------------\n");
  printf("Will process %d image(s) with %d iteration(s) each\n", numFiles, iterations);

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Process each input/output pair
  for (int fileIdx = 0; fileIdx < numFiles; fileIdx++) {
      int inputArgIdx = 2 + fileIdx * 2;
      int outputArgIdx = 3 + fileIdx * 2;
      
      printf("\nProcessing image %d of %d: %s -> %s\n", fileIdx + 1, numFiles, argv[inputArgIdx], argv[outputArgIdx]);

      // Load the image
      int width, height, channels;
      unsigned char *imageIn = stbi_load(argv[inputArgIdx], &width, &height, &channels, 0);

      if (!imageIn) {
          printf("Error loading image file: %s\n", stbi_failure_reason());
          continue; // Skip to next image
      }

      printf("Loaded image: %s (%dx%d with %d channels)\n", argv[inputArgIdx], width, height, channels);

      // Allocate GPU memory
      size_t imageSize = (size_t)height * width * channels * sizeof(unsigned char);
      unsigned char *d_buffer1 = NULL, *d_buffer2 = NULL;
      CHECK_CUDA(cudaMalloc((void **)&d_buffer1, imageSize));
      CHECK_CUDA(cudaMalloc((void **)&d_buffer2, imageSize));
      
      // Copy image to GPU
      CHECK_CUDA(cudaMemcpy(d_buffer1, imageIn, imageSize, cudaMemcpyHostToDevice));

      // Set up grid and block dimensions
      dim3 block(WORKGROUP_SIZE, WORKGROUP_SIZE);
      dim3 grid((width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE, (height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);

      // Allocate histogram and minimum value memory once
      unsigned int *d_histogram;
      unsigned int *d_minimum;
      CHECK_CUDA(cudaMalloc((void **)&d_histogram, BINS * sizeof(unsigned int)));
      CHECK_CUDA(cudaMalloc((void **)&d_minimum, sizeof(unsigned int)));

      // Run the algorithm n times
      for (int iter = 0; iter < iterations; iter++) {
          printf("\nIteration %d of %d:\n", iter + 1, iterations);
          
          // RGB to YUV conversion
          printf("Performing RGB to YUV conversion...\n");
          cudaEventRecord(start);
          rgb_to_yuv<<<grid, block>>>(d_buffer1, d_buffer2, width, height);
          cudaEventRecord(stop);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime(&milliseconds, start, stop);
          printf("RGB to YUV conversion complete. Time taken: %.3f ms\n", milliseconds);

          // Histogram calculation
          printf("Calculating histogram...\n");
          CHECK_CUDA(cudaMemset(d_histogram, 0, BINS * sizeof(unsigned int)));
          cudaEventRecord(start);
          histogramKernel<<<grid, block>>>(d_buffer2, d_histogram, width, height);
          cudaEventRecord(stop);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime(&milliseconds, start, stop);
          printf("Histogram calculation complete. Time taken: %.3f ms\n", milliseconds);

          // Cumulative histogram calculation
          printf("Calculating cumulative histogram...\n");
          CHECK_CUDA(cudaMemset(d_minimum, 0, sizeof(unsigned int)));
          cudaEventRecord(start);
          computeCumulativeHistogram<<<1, 1>>>(d_histogram, d_minimum);
          cudaEventRecord(stop);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime(&milliseconds, start, stop);
          printf("Cumulative histogram calculation complete. Time taken: %.3f ms\n", milliseconds);

          // Histogram equalization
          printf("Performing histogram equalization...\n");
          cudaEventRecord(start);
          histogramEqualizationKernel<<<grid, block>>>(d_buffer2, d_histogram, width, height, d_minimum);
          cudaEventRecord(stop);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime(&milliseconds, start, stop);
          printf("Histogram equalization complete. Time taken: %.3f ms\n", milliseconds);

          // YUV to RGB conversion
          printf("Performing YUV to RGB conversion...\n");
          cudaEventRecord(start);
          yuv_to_rgb<<<grid, block>>>(d_buffer2, d_buffer1, width, height);
          cudaEventRecord(stop);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime(&milliseconds, start, stop);
          printf("YUV to RGB conversion complete. Time taken: %.3f ms\n", milliseconds);
      }

      // Copy final result back to host
      printf("Copying final image from GPU to CPU...\n");
      CHECK_CUDA(cudaMemcpy(imageIn, d_buffer1, imageSize, cudaMemcpyDeviceToHost));
      printf("Copy complete.\n");

      // Save final image
      printf("Saving final processed image to %s...\n", argv[outputArgIdx]);
      if (stbi_write_png(argv[outputArgIdx], width, height, channels, imageIn, width * channels) == 0) {
          printf("Error writing output image file: %s\n", argv[outputArgIdx]);
      } else {
          printf("Successfully processed and saved image to %s\n", argv[outputArgIdx]);
      }

      // Cleanup resources for this image
      CHECK_CUDA(cudaFree(d_buffer1));
      CHECK_CUDA(cudaFree(d_buffer2));
      CHECK_CUDA(cudaFree(d_histogram));
      CHECK_CUDA(cudaFree(d_minimum));
      stbi_image_free(imageIn);
  }

  // Final cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printf("\n-------------- GPU compute finished --------------\n");

  return 0;
}