#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Include stb_image headers
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define BINS 256
#define ARGS_LEN (2)

typedef struct histogram {
  unsigned int *R;
  unsigned int *G;
  unsigned int *B;
} histogram;

typedef struct histogramYUV {
  unsigned int *Y;
  unsigned int *U;
  unsigned int *V;
} histogramYUV;

void rgb_to_yuv(unsigned char *pixel) {
  // Get RGB values
  unsigned char R = pixel[0];
  unsigned char G = pixel[1];
  unsigned char B = pixel[2];

  // Convert to YUV using standard conversion formulas
  unsigned char Y =
      static_cast<unsigned char>(0.299f * R + 0.587f * G + 0.114f * B);
  unsigned char U = static_cast<unsigned char>(-0.168736f * R - 0.331264f * G +
                                               0.5f * B + 128);
  unsigned char V = static_cast<unsigned char>(0.5f * R - 0.418688f * G -
                                               0.081312f * B + 128);

  // Store YUV values back in place of RGB
  pixel[0] = Y;
  pixel[1] = U;
  pixel[2] = V;
  // printf("Y: %d\n", Y);
}

void yuv_to_rgb(unsigned char *pixel) {
  // Get YUV values
  unsigned char Y = pixel[0];
  unsigned char U = pixel[1];
  unsigned char V = pixel[2];

  // Adjust U and V by subtracting 128
  float U_adj = U - 128.0f;
  float V_adj = V - 128.0f;

  // Convert to RGB using standard conversion formulas
  int R = static_cast<int>(Y + 1.402f * V_adj);
  int G = static_cast<int>(Y - 0.344136f * U_adj - 0.714136f * V_adj);
  int B = static_cast<int>(Y + 1.772f * U_adj);

  // Clamp values to valid range [0, 255]
  R = std::max(0, std::min(255, R));
  G = std::max(0, std::min(255, G));
  B = std::max(0, std::min(255, B));

  // Store RGB values back in place of YUV
  pixel[0] = static_cast<unsigned char>(R);
  pixel[1] = static_cast<unsigned char>(G);
  pixel[2] = static_cast<unsigned char>(B);
}

// Function to convert an entire image from RGB to YUV
void rgb_to_yuv_image(unsigned char *image, int width, int height) {
  for (int i = 0; i < width * height * 3; i += 3) {
    rgb_to_yuv(&image[i]);
  }
}

// Function to convert an entire image from YUV to RGB
void yuv_to_rgb_image(unsigned char *image, int width, int height) {
  for (int i = 0; i < width * height * 3; i += 3) {
    yuv_to_rgb(&image[i]);
  }
}
// -----------------------------------------------------------------------------
// Helper function to print the histogram (for debugging)
// -----------------------------------------------------------------------------
void printHistogram(histogram H) {
  printf("Colour\tNo. Pixels\n");
  for (int i = 0; i < BINS; i++) {
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
void drawColumn(unsigned char *image, int width, int height, int i, int value,
                int max, int offset) {
  int calc_height = height - (int)(((float)value / (float)max) * 255) - 1;
  for (int j = calc_height; j < height; j++) {
    int pos = j * width * 4 + i * 4;
    for (int k = 0; k < 3; k++) {
      if (k == offset)
        continue;
      if (j == calc_height)
        image[pos + k] = 0;
      else
        image[pos + k] = image[pos + k] / 2;
    }
  }
}

void drawHistogram(histogramYUV H, int argWidth, int argHeight) {
  const int width = BINS;
  const int height = BINS;
  unsigned char *image =
      (unsigned char *)malloc(height * width * 4 * sizeof(unsigned char));

  // Initialize to white
  for (int i = 0; i < height * width * 4; i++)
    image[i] = 255;

  // Find the maximum value in the histogram
  int max = 255;
  for (int i = 0; i < BINS; i++) {
    if (H.Y[i] > max)
      max = H.Y[i];
    if (H.U[i] > max)
      max = H.U[i];
    if (H.V[i] > max)
      max = H.V[i];
  }

  for (int i = 0; i < BINS; i++) {
    if (H.Y[i])
      drawColumn(image, width, height, i, H.Y[i], max, 0);
    if (H.U[i])
      drawColumn(image, width, height, i, H.U[i], max, 2);
    if (H.V[i])
      drawColumn(image, width, height, i, H.V[i], max, 1);
  }

  // Save the histogram as PNG using stb_image_write
  char f_out_name[256];
  printf("> Saving the Histogram as PNG in 'gpu-out/%d-%d-gpu-hist.png'\n",
         argWidth, argHeight);
  snprintf(f_out_name, sizeof(f_out_name), "cpu-out/%d-%d-gpu-hist.png",
           argWidth, argHeight);

  // Make sure the directory exists
  system("mkdir -p cpu-out");

  // Save the image using stb_image_write
  stbi_write_png(f_out_name, width, height, 4, image, width * 4);
  free(image);
}

histogramYUV createHistogram(unsigned char *image, int width, int height) {
  histogramYUV hist;
  hist.Y = (uint *)calloc(BINS, sizeof(uint));
  hist.U = (uint *)calloc(BINS, sizeof(uint));
  hist.V = (uint *)calloc(BINS, sizeof(uint));

  if (!image || width <= 0 || height <= 0) {
    return hist; // return empty histogram for invalid input
  }

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      size_t offset = (y * width + x) * 3; // 3 bytes per pixel (Y,U,V)

      // Safety check
      if (offset + 2 >= (size_t)width * height * 3) {
        break;
      }

      // Get YUV values
      unsigned char Y = image[offset];
      unsigned char U = image[offset + 1];
      unsigned char V = image[offset + 2];

      // increment histogram
      hist.Y[Y]++;
      hist.U[U]++;
      hist.V[V]++;
    }
  }

  return hist;
}

void inplaceCumulative(uint *buffer, int size) {
  for (int i = 1; i < size; i++) {
    int prev = buffer[i - 1];
    // printf("buff: %d\n", buffer[i]);
    buffer[i] += prev;
  }
}

void inplaceNormalizeLuminance(unsigned char *image, int width, int height,
                               uint *cumulativeHist) {
  if (!image || !cumulativeHist || width <= 0 || height <= 0) {
    return; // Invalid input
  }

  const int L = 256; // Number of intensity levels (for 8-bit images)
  int totalPixels = width * height;

  // Step 1: Find min non-zero value in cumulative histogram
  uint minCdf = 0;
  for (int i = 0; i < L; i++) {
    if (cumulativeHist[i] != 0) {
      minCdf = cumulativeHist[i];
      break;
    }
  }

  // Step 2: Apply histogram equalization to Y channel
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int offset = (y * width + x) *
                   3; // YUV format (Y at offset, U at offset+1, V at offset+2)

      // Get original luminance (Y)
      unsigned char Y_original = image[offset];

      // Compute new Y using the exact formula from the reference
      uint newY = (uint)((cumulativeHist[Y_original] - minCdf) * (L - 1) /
                             (float)(totalPixels - minCdf) +
                         0.5f); // +0.5 for rounding

      // Clamp to [0, 255]
      if (newY > 255)
        newY = 255;

      // Update only the Y channel (preserve U and V)
      image[offset] = (unsigned char)newY;
    }
  }
}

// Helper function to get current time in milliseconds
static double get_time_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (ts.tv_sec * 1000.0) + (ts.tv_nsec / 1000000.0);
}

int main(int argc, char **argv) {
  if (argc != ARGS_LEN + 1) {
    printf("Invalid arguments ~ Requires image path!\n");
    return -1;
  }

  printf("-------------- Starting CPU compute --------------\n");
  double total_start = get_time_ms();

  // 1. Load image
  double load_start = get_time_ms();
  int width, height, channels;
  unsigned char *imageIn = stbi_load(argv[1], &width, &height, &channels, 0);
  if (!imageIn) {
    printf("Error loading image file: %s\n", stbi_failure_reason());
    return -1;
  }
  double load_end = get_time_ms();
  printf("Loaded image: %dx%d with %d channels (%.4f ms)\n", width, height,
         channels, load_end - load_start);

  // 2. Convert RGB to YUV
  double convert_start = get_time_ms();
  rgb_to_yuv_image(imageIn, width, height);
  double convert_end = get_time_ms();
  printf("RGB→YUV conversion: %.4f ms\n", convert_end - convert_start);

  // 3. Create histogram
  double hist_start = get_time_ms();
  histogramYUV hist = createHistogram(imageIn, width, height);
  double hist_end = get_time_ms();
  printf("Histogram creation: %.4f ms\n", hist_end - hist_start);

  // 4. Cumulative histogram
  double cum_start = get_time_ms();
  inplaceCumulative(hist.Y, BINS);
  double cum_end = get_time_ms();
  printf("Cumulative histogram: %.4f ms\n", cum_end - cum_start);

  // 5. Normalize luminance
  double norm_start = get_time_ms();
  inplaceNormalizeLuminance(imageIn, width, height, hist.Y);
  double norm_end = get_time_ms();
  printf("Luminance normalization: %.4f ms\n", norm_end - norm_start);

  // 6. Convert back to RGB
  double backconv_start = get_time_ms();
  yuv_to_rgb_image(imageIn, width, height);
  double backconv_end = get_time_ms();
  printf("YUV→RGB conversion: %.4f ms\n", backconv_end - backconv_start);

  // 7. Save output
  double save_start = get_time_ms();
  int write_result = stbi_write_png(argv[2], width, height, channels, imageIn,
                                    width * channels);
  double save_end = get_time_ms();
  printf("Image saving: %.4f ms\n", save_end - save_start);

  // Free resources
  stbi_image_free(imageIn);
  free(hist.Y);
  free(hist.U);
  free(hist.V);

  double total_end = get_time_ms();
  printf("\nTotal execution time: %.4f ms\n", total_end - total_start);
  printf("Total execution time per pixel: %.4f ms\n",
         (total_end - total_start) / (width * height));
  printf("-------------- Processing complete --------------\n");

  return 0;
}