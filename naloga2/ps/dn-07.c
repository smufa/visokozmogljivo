#include "FreeImage.h"
#include <CL/cl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define SIZE (1024)
#define WORKGROUP_SIZE (16)
#define MAX_SOURCE_SIZE (16384)
#define DEBUG (0)
#define BINS 256
#define ARGS_LEN (1)

typedef struct histogram
{
    unsigned int *R;
    unsigned int *G;
    unsigned int *B;
} histogram;

void drawColumn(unsigned char *image, int width, int height, int i, int value, int max, int offset);
void drawHistogram(histogram H, int argWwidth, int argHeight);
void printHistogram(histogram H);

int main(int argc, char **argv)
{

    if (argc != ARGS_LEN + 1)
    {
        printf("Invalid arguments ~ Requires image path! \n");
        return -1;
    }

    printf("-------------- Starting GPU compute --------------\n");

    // Load image from file
    FIBITMAP *imageBitmap = FreeImage_Load(FIF_JPEG, argv[1], 0);
    // Convert it to a 32-bit image
    FIBITMAP *imageBitmap32 = FreeImage_ConvertTo32Bits(imageBitmap);

    // Get image dimensions
    int width = FreeImage_GetWidth(imageBitmap32);
    int height = FreeImage_GetHeight(imageBitmap32);
    int pitch = FreeImage_GetPitch(imageBitmap32);

    // printf("pitch: %d \n", pitch);

    // Preapare room for a raw data copy of the image
    unsigned char *imageIn = (unsigned char *)malloc(height * width * sizeof(unsigned char) * 4);

    // Initalize the histogram
    histogram H;
    H.B = (unsigned int *)calloc(BINS, sizeof(unsigned int));
    H.G = (unsigned int *)calloc(BINS, sizeof(unsigned int));
    H.R = (unsigned int *)calloc(BINS, sizeof(unsigned int));

    // Extract raw data from the image
    FreeImage_ConvertToRawBits(imageIn, imageBitmap, pitch, 32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);

    // Free source image data
    FreeImage_Unload(imageBitmap32);
    FreeImage_Unload(imageBitmap);

    unsigned int *t = (unsigned int *)calloc(BINS * 3, sizeof(unsigned int));

    char ch;
    int i, j;
    cl_int ret;
    FILE *fp;
    char *source_str;
    size_t source_size;

    // Read the file
    fp = fopen("kernel.cl", "r");
    if (!fp)
    {
        fprintf(stderr, "File missing?\n");
        return 1;
    }
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    source_str[source_size] = '\0';
    fclose(fp);

    // Platform data and info
    cl_platform_id platform_id[10];
    cl_uint ret_num_platforms;
    char *buf;
    size_t buf_len;
    ret = clGetPlatformIDs(10, platform_id, &ret_num_platforms);

    // GPU data and info
    cl_device_id device_id[10];
    cl_uint ret_num_devices;
    ret = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 10, device_id, &ret_num_devices);

    // Create the context
    cl_context context = clCreateContext(NULL, 1, &device_id[0], NULL, NULL, &ret);

    // Command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id[0], 0, &ret);

    // Split the work ~ 2D
    size_t local_item_size[2] = {WORKGROUP_SIZE, WORKGROUP_SIZE};
    size_t num_groups[2] = {((height - 1) / local_item_size[0] + 1), ((width - 1) / local_item_size[1] + 1)};
    size_t global_item_size[2] = {num_groups[0] * local_item_size[0], num_groups[1] * local_item_size[1]};

    // Allocate the memory on the GPU. We only allocate the space for the image and histogram
    cl_mem img_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, height * width * sizeof(unsigned char) * 4, imageIn, &ret);
    cl_mem h_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, BINS * 3 * sizeof(unsigned int), t, &ret);

    // Create the program
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, NULL, &ret);

    // Compile
    ret = clBuildProgram(program, 1, &device_id[0], NULL, NULL, NULL);

    // Log
    size_t build_log_len;
    char *build_log;
    ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
    build_log = (char *)malloc(sizeof(char) * (build_log_len + 1));
    ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, build_log_len, build_log, NULL);
    printf("%s\n", build_log);
    free(build_log);
    if (build_log_len > 2)
        return 1;

    // Create the Kernel
    cl_kernel kernel = clCreateKernel(program, "histogramkernel", &ret);

    // Kernel arguments
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&img_mem_obj);
    ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&h_mem_obj);
    ret |= clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&width);
    ret |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&height);

    // Start the initial timer
    double start_t = omp_get_wtime();

    // Start
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_item_size, local_item_size, 0, NULL, NULL);

    double write_end_t = omp_get_wtime() - start_t;

    // Read
    ret = clEnqueueReadBuffer(command_queue, h_mem_obj, CL_TRUE, 0, BINS * sizeof(unsigned int), H.R, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, h_mem_obj, CL_TRUE, BINS * sizeof(unsigned int), BINS * sizeof(unsigned int), H.G, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, h_mem_obj, CL_TRUE, BINS * 2 * sizeof(unsigned int), BINS * sizeof(unsigned int), H.B, 0, NULL, NULL);

    double write_calc_end_t = omp_get_wtime() - start_t;
    double calc_end_t = omp_get_wtime() - start_t - write_end_t;

    // Cleaning up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(img_mem_obj);
    ret = clReleaseMemObject(h_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    // H.R = t;

    double end_t = omp_get_wtime() - start_t;

    printf("> Data transfer:   %lfs \n", write_end_t);
    printf("> Calculations:    %lfs \n", calc_end_t);
    printf("> Data + Calc:     %lfs \n", write_calc_end_t);
    printf("> Start - finish:  %lfs \n", end_t);

    drawHistogram(H, width, height);
    // printHistogram(H);

    return 0;
}

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
    int pitch = ((32 * width + 31) / 32) * 4;

    unsigned char *image =
        (unsigned char *)malloc(height * width * sizeof(unsigned char) * 4);

    // set everything to white
    for (int i = 0; i < height * width * 4; i++)
        image[i] = 255;

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

    FIBITMAP *dst = FreeImage_ConvertFromRawBits(
        image, width, height, pitch, 32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK,
        FI_RGBA_BLUE_MASK, TRUE);

    char f_out_name[256];

    printf("> Saving the Histogram as PNG in 'gpu-out/%d-%d-gpu-hist.png'\n", argWidth, argHeight);

    snprintf(f_out_name, sizeof(f_out_name), "gpu-out/%d-%d-gpu-hist.png", argWidth, argHeight);

    FreeImage_Save(FIF_PNG, dst, f_out_name, 0);
}

/*
------------------------------------ 640x480 
CPU:
> CPU compute took:   0.000877s 
GPU:
> Data transfer:   0.000564s 
> Calculations:    0.000265s 
> Data + Calc:     0.000829s 
S:
1.05790109

------------------------------------ 800x600
CPU:
> CPU compute took:   0.000877s
GPU:
> Data transfer:   0.000727s 
> Calculations:    0.000653s 
> Data + Calc:     0.001380s 
S:
0.635507246

------------------------------------ 1600X900
CPU:
> CPU compute took:   0.002226s 
GPU:
> Data transfer:   0.001342s 
> Calculations:    0.001253s 
> Data + Calc:     0.002595s 
S:
0.857803468

------------------------------------ 1920x1080
CPU:
> CPU compute took:   0.003008s 
GPU:
> Data transfer:   0.001704s 
> Calculations:    0.001559s 
> Data + Calc:     0.003263s 
S:
0.921851057

------------------------------------ 3840x2160
CPU:
> CPU compute took:   0.012127s 
GPU:
> Data transfer:   0.005062s 
> Calculations:    0.005202s 
> Data + Calc:     0.010264s 
S:
1.18150818

------------------------------------ OPOMBE
Sami izračuni so bili hitrejši, vendar prenašanje slike v globalni pomnilnik bistveno upočasni delovanje.
Prav tako slike, ki imajo nekoliko bolj enakomerne barve (so manj barvite) upočasnijo delovanje, ker 
pride do več zaklepanj pri `atomic_add` in `atomic_inc`.
Večje slike so hitrejše na GPU.
*/