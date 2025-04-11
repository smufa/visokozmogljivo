#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

// Include stb_image headers
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define SIZE (1024)
#define WORKGROUP_SIZE (16)
#define BINS 256
#define ARGS_LEN (1)

typedef struct histogram {
    unsigned int *R;
    unsigned int *G;
    unsigned int *B;
} histogram;

void drawColumn(unsigned char *image, int width, int height, int i, int value, int max, int offset);
void drawHistogram(histogram H, int argWidth, int argHeight);
void printHistogram(histogram H);

// -----------------------------------------------------------------------------
// CUDA kernel (converted from OpenCL kernel)
// -----------------------------------------------------------------------------
__global__ void histogramKernel(unsigned char *image, unsigned int *H, int width, int height, int channels)
{
    // Each block uses a shared buffer of size 256*3.
    // Here we assume each block is 16x16 = 256 threads, mapping 1-to-1.
    __shared__ unsigned int buffer[256 * 3];
    
    // Compute local (block) indices
    int l_i = threadIdx.x;
    int l_j = threadIdx.y;
    int l_s = blockDim.x;  // Assumes blockDim.x == blockDim.y == WORKGROUP_SIZE

    // Global indices: (global_i = row, global_j = column)
    int global_i = blockIdx.x * blockDim.x + l_i;
    int global_j = blockIdx.y * blockDim.y + l_j;
    
    // Compute a linear index for the thread within the block.
    int shared_idx = l_i * l_s + l_j;
    
    // Initialize shared memory. (There are 3 segments: for channels 1, 2 and 3)
    if (shared_idx < BINS) {
        buffer[shared_idx]         = 0;
        buffer[shared_idx + BINS]    = 0;
        buffer[shared_idx + 2 * BINS] = 0;
    }
    __syncthreads();

    // Process only if within image bounds
    if (global_i < height && global_j < width) {
        // The image is stored as channels per pixel (e.g., RGB or RGBA)
        int offset = (global_i * width + global_j) * channels;

        // Note: The original OpenCL code uses atomic_inc.
        // In CUDA, we use atomicAdd (with value 1) to update the shared histogram.
        if (channels >= 3) {
            atomicAdd(&buffer[image[offset + 2]], 1);             // Blue channel
            atomicAdd(&buffer[image[offset + 1] + BINS], 1);      // Green channel
            atomicAdd(&buffer[image[offset] + 2 * BINS], 1);      // Red channel
        } else if (channels == 1) {
            // For grayscale images, use the same value for all channels
            atomicAdd(&buffer[image[offset]], 1);                 // Blue channel
            atomicAdd(&buffer[image[offset] + BINS], 1);          // Green channel
            atomicAdd(&buffer[image[offset] + 2 * BINS], 1);      // Red channel
        }
    }
    
    __syncthreads();
    
    // Each thread in the block updates the global histogram.
    if (shared_idx < BINS) {
        atomicAdd(&H[shared_idx], buffer[shared_idx]);
        atomicAdd(&H[shared_idx + BINS], buffer[shared_idx + BINS]);
        atomicAdd(&H[shared_idx + 2 * BINS], buffer[shared_idx + 2 * BINS]);
    }
}

// -----------------------------------------------------------------------------
// Host main() using CUDA runtime API
// -----------------------------------------------------------------------------
int main(int argc, char **argv)
{
    if (argc != ARGS_LEN + 1) {
        printf("Invalid arguments ~ Requires image path!\n");
        return -1;
    }

    printf("-------------- Starting GPU compute (CUDA) --------------\n");

    // 1. Load image using stb_image
    int width, height, channels;
    unsigned char *imageIn = stbi_load(argv[1], &width, &height, &channels, 0);
    
    if (!imageIn) {
        printf("Error loading image file: %s\n", stbi_failure_reason());
        return -1;
    }
    
    printf("Loaded image: %dx%d with %d channels\n", width, height, channels);

    // Initialize histogram arrays for R, G, B channels.
    histogram H;
    H.R = (unsigned int *)calloc(BINS, sizeof(unsigned int));
    H.G = (unsigned int *)calloc(BINS, sizeof(unsigned int));
    H.B = (unsigned int *)calloc(BINS, sizeof(unsigned int));

    // Allocate device memory
    unsigned char *d_image = NULL;
    unsigned int *d_H = NULL;
    size_t imageSize = height * width * channels * sizeof(unsigned char);
    size_t histSize  = BINS * 3 * sizeof(unsigned int);

    cudaMalloc((void**)&d_image, imageSize);
    cudaMalloc((void**)&d_H, histSize);

    // Copy image data to device; also initialize histogram memory on device to 0.
    cudaMemcpy(d_image, imageIn, imageSize, cudaMemcpyHostToDevice);
    cudaMemset(d_H, 0, histSize);

    // Setup execution dimensions
    dim3 block(WORKGROUP_SIZE, WORKGROUP_SIZE);
    // Grid dimensions: enough blocks to cover all rows and columns.
    dim3 grid((height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE,
              (width  + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);

    // With CUDA event-based timing:
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    histogramKernel<<<grid, block>>>(d_image, d_H, width, height, channels);
    cudaEventRecord(stop);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    // Calculate elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Convert to seconds for consistent output format
    float execution_time = milliseconds / 1000.0f;

    printf("> Computation (kernel launch + execution) took: %lfs \n", execution_time);

    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy histogram data back from the device.
    // The global memory layout is: first 256 integers for channel1, next 256 for channel2, next 256 for channel3.
    cudaMemcpy(H.R, d_H, BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(H.G, d_H + BINS, BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(H.B, d_H + 2 * BINS, BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Cleanup device memory
    cudaFree(d_image);
    cudaFree(d_H);

    // Draw and save the histogram image
    drawHistogram(H, width, height);
    // Optionally, you can call printHistogram(H);

    // Cleanup host memory for histogram and image data.
    stbi_image_free(imageIn);
    free(H.R);
    free(H.G);
    free(H.B);

    return 0;
}

// -----------------------------------------------------------------------------
// Helper function to print the histogram (for debugging)
// -----------------------------------------------------------------------------
void printHistogram(histogram H)
{
    printf("Colour\tNo. Pixels\n");
    for (int i = 0; i < BINS; i++)
    {
        if (H.B[i] > 0)
            printf("%dB\t%d\n", i, H.B[i]);
        if (H.G[i] > 0)
            printf("%dG\t%d\n", i, H.G[i]);
        if (H.R[i] > 0)
            printf("%dR\t%d\n", i, H.R[i]);
    }
}

// -----------------------------------------------------------------------------
// Drawing routines updated to use stb_image_write
// -----------------------------------------------------------------------------
void drawColumn(unsigned char *image, int width, int height, int i, int value, int max, int offset)
{
    int calc_height = height - (int)(((float)value / (float)max) * 255) - 1;
    for (int j = calc_height; j < height; j++)
    {
        int pos = j * width * 4 + i * 4;
        for (int k = 0; k < 3; k++)
        {
            if (k == offset)
                continue;
            if (j == calc_height)
                image[pos + k] = 0;
            else
                image[pos + k] = image[pos + k] / 2;
        }
    }
}

void drawHistogram(histogram H, int argWidth, int argHeight)
{
    const int width = BINS;
    const int height = BINS;
    unsigned char *image = (unsigned char *)malloc(height * width * 4 * sizeof(unsigned char));

    // Initialize to white
    for (int i = 0; i < height * width * 4; i++)
        image[i] = 255;
    
    // Find the maximum value in the histogram
    int max = 255;
    for (int i = 0; i < BINS; i++)
    {
        if (H.B[i] > max)
            max = H.B[i];
        if (H.G[i] > max)
            max = H.G[i];
        if (H.R[i] > max)
            max = H.R[i];
    }

    for (int i = 0; i < BINS; i++)
    {
        if (H.B[i])
            drawColumn(image, width, height, i, H.B[i], max, 0);
        if (H.R[i])
            drawColumn(image, width, height, i, H.R[i], max, 2);
        if (H.G[i])
            drawColumn(image, width, height, i, H.G[i], max, 1);
    }

    // Save the histogram as PNG using stb_image_write
    char f_out_name[256];
    printf("> Saving the Histogram as PNG in 'gpu-out/%d-%d-gpu-hist.png'\n", argWidth, argHeight);
    snprintf(f_out_name, sizeof(f_out_name), "gpu-out/%d-%d-gpu-hist.png", argWidth, argHeight);
    
    // Make sure the directory exists
    system("mkdir -p gpu-out");
    
    // Save the image using stb_image_write
    stbi_write_png(f_out_name, width, height, 4, image, width * 4);

    free(image);
}
