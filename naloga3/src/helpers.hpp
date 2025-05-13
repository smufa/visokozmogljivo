#ifndef GRAY_SCOTT_HELPERS_HPP
#define GRAY_SCOTT_HELPERS_HPP

#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"
#include <cstddef>
#include <iostream>

// Function to convert unsigned char array to float array
void ucharToFloat(const unsigned char* uchar_data, float* float_data, size_t size);

// Function to convert float array to unsigned char array
void floatToUchar(const float* float_data, unsigned char* uchar_data, size_t size);

// Function to load a PNG image and return a float array
float* loadImage(const char* filename, int* width, int* height, int* channels) {
    int w, h, n;
    unsigned char* data = stbi_load(filename, &w, &h, &n, 0);
    if (data == nullptr) {
        std::cerr << "Error loading image: " << filename << std::endl;
        return nullptr;
    }

    *width = w;
    *height = h;
    *channels = n;

    float* float_data = new float[w * h * n];
    ucharToFloat(data, float_data, w * h * n);

    stbi_image_free(data);
    return float_data;
}

// Function to write a float array to a PNG image
bool writeImage(const char* filename, int width, int height, int channels, const float* float_data) {
    unsigned char* uchar_data = new unsigned char[width * height * channels];
    floatToUchar(float_data, uchar_data, width * height * channels);

    int stride_in_bytes = width * channels;
    int success = stbi_write_png(filename, width, height, channels, uchar_data, stride_in_bytes);

    delete[] uchar_data;
    return success != 0;
}

void ucharToFloat(const unsigned char* uchar_data, float* float_data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        float_data[i] = static_cast<float>(uchar_data[i]) / 255.0f;
    }
}

void floatToUchar(const float* float_data, unsigned char* uchar_data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        uchar_data[i] = static_cast<unsigned char>(float_data[i] * 255.0f);
    }
}

// Function to access image data without bounds checking
float& imageAt(float* image_data, int width, int height, int x, int y, int channel) {
    x = (x % width + width) % width;
    y = (y % height + height) % height;
    size_t index = (y * width + x) * 2 + channel;
    return image_data[index];
}

#endif