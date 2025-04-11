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

__global__ void rgb_to_yuv(unsigned char *image, int width, int height) {
    // Calculate pixel position
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if within image bounds
    if (x < width && y < height) {
        // Calculate the offset in the image array
        int offset = (y * width + x) * 3; // 3 bytes per pixel (R,G,B)
        
        // Get RGB values
        unsigned char R = image[offset];
        unsigned char G = image[offset + 1];
        unsigned char B = image[offset + 2];
        
        // Convert to YUV using the matrix from the formula
        // [Y]   [  0.299      0.587      0.114   ] [R]   [ 0   ]
        // [U] = [ -0.168736  -0.331264   0.5     ] [G] + [ 128 ]
        // [V]   [  0.5       -0.418688  -0.081312] [B]   [ 128 ]
        
        // Calculate YUV values
        unsigned char Y = (unsigned char)(0.299f * R + 0.587f * G + 0.114f * B);
        unsigned char U = (unsigned char)(-0.168736f * R - 0.331264f * G + 0.5f * B + 128);
        unsigned char V = (unsigned char)(0.5f * R - 0.418688f * G - 0.081312f * B + 128);
        
        // Store YUV values back in place of RGB
        image[offset] = Y;
        image[offset + 1] = U;
        image[offset + 2] = V;
    }
}

__global__ void yuv_to_rgb(unsigned char *image, int width, int height) {
    // Calculate pixel position
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if within image bounds
    if (x < width && y < height) {
        // Calculate the offset in the image array
        int offset = (y * width + x) * 3; // 3 bytes per pixel (Y,U,V)
        
        // Get YUV values
        unsigned char Y = image[offset];
        unsigned char U = image[offset + 1];
        unsigned char V = image[offset + 2];
        
        // Adjust U and V by subtracting 128
        float U_adj = U - 128.0f;
        float V_adj = V - 128.0f;
        
        // Convert to RGB using the matrix from the formula
        // [R]   [1      0          1.402   ] [  Y  ]
        // [G] = [1  -0.344136  -0.714136   ] [U-128]
        // [B]   [1    1.772        0       ] [V-128]
        
        // Calculate RGB values and clamp to [0, 255]
        int R = Y + 1.402f * V_adj;
        int G = Y - 0.344136f * U_adj - 0.714136f * V_adj;
        int B = Y + 1.772f * U_adj;
        
        // Clamp values to valid range [0, 255]
        R = max(0, min(255, R));
        G = max(0, min(255, G));
        B = max(0, min(255, B));
        
        // Store RGB values back in place of YUV
        image[offset] = (unsigned char)R;
        image[offset + 1] = (unsigned char)G;
        image[offset + 2] = (unsigned char)B;
    }
}


// -----------------------------------------------------------------------------
// CUDA kernel (converted from OpenCL kernel)
// -----------------------------------------------------------------------------
__global__ void histogramKernel(unsigned char *image, unsigned int *H, int width, int height, int channel)
{
    // Each block uses a shared buffer of size 256*3.
    // Here we assume each block is 16x16 = 256 threads, mapping 1-to-1.
    __shared__ unsigned int buffer[256];
    
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
    }
    __syncthreads();

    // Process only if within image bounds
    if (global_i < height && global_j < width) {
        // The image is stored as channels per pixel (e.g., RGB or RGBA)
        int offset = (global_i * width + global_j);

        // Note: The original OpenCL code uses atomic_inc.
        // In CUDA, we use atomicAdd (with value 1) to update the shared histogram.
        // For grayscale images, use the same value for all channels
        atomicAdd(&buffer[image[offset]], 1);                 // Blue channel
    }
    
    __syncthreads();
    
    // Each thread in the block updates the global histogram.
    if (shared_idx < BINS) {
        atomicAdd(&H[shared_idx], buffer[shared_idx]);
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
