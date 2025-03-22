#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Only include STB implementations in one translation unit
#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif
#include "stb_image.h"

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#include "stb_image_write.h"

class Image {
private:
  std::vector<float> data;
  int width, height, channels;

  // Helper function to get file extension
  static std::string get_extension(const std::string &filename) {
    size_t dot = filename.find_last_of(".");
    if (dot == std::string::npos)
      return "";
    std::string ext = filename.substr(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext;
  }

public:
  // Constructor for empty image
  Image() : width(0), height(0), channels(0) {}

  // Constructor with data
  Image(const std::vector<float> &img_data, int w, int h, int c)
      : data(img_data), width(w), height(h), channels(c) {}

  // Constructor for creating an image of specified size with zeros
  Image(int w, int h, int c) : width(w), height(h), channels(c) {
    data.resize(w * h * c, 0.0f);
  }

  // Load image from file
  static Image load(const std::string &filename) {
    int width, height, channels;
    unsigned char *img_data =
        stbi_load(filename.c_str(), &width, &height, &channels, 0);

    if (!img_data) {
      throw std::runtime_error(std::string("Error loading image: ") +
                               stbi_failure_reason());
    }

    // Convert unsigned char data to float (normalizing to 0.0-1.0 range)
    std::vector<float> data_vec(width * height * channels);
    for (int i = 0; i < width * height * channels; i++) {
      data_vec[i] = static_cast<float>(img_data[i]) / 255.0f;
    }

    Image result(data_vec, width, height, channels);

    stbi_image_free(img_data);
    return result;
  }

  // Save image to file
  bool save(const std::string &filename) const {
    if (data.empty()) {
      std::cerr << "Cannot save empty image" << std::endl;
      return false;
    }

    // Convert float data back to unsigned char for saving
    std::vector<unsigned char> temp_data(width * height * channels);
    for (size_t i = 0; i < data.size(); i++) {
      // Clamp values to 0.0-1.0 range and convert to 0-255
      float clamped = std::max(0.0f, std::min(1.0f, data[i]));
      temp_data[i] = static_cast<unsigned char>(clamped * 255.0f);
    }

    std::string ext = get_extension(filename);
    int success = 0;

    if (ext == "png") {
      success = stbi_write_png(filename.c_str(), width, height, channels,
                               temp_data.data(), width * channels);
    } else if (ext == "jpg" || ext == "jpeg") {
      success = stbi_write_jpg(filename.c_str(), width, height, channels,
                               temp_data.data(), 90);
    } else {
      std::cerr << "Unsupported format. Use .png or .jpg" << std::endl;
      return false;
    }

    return success != 0;
  }

  // Indexing function (channel, x, y)
  float &at(int c, int x, int y) {
    // Check if channel is valid
    if (c < 0 || c >= channels) {
      throw std::out_of_range("Channel index out of range");
    }

    // if (y > height)
    // {
    //     printf("y: %d is over height\n");
    // }
    // if (y < 0)
    // {
    //     printf("y: %d is under 0\n");
    // }
    // Clamp x and y coordinates to valid image boundaries
    int nx = std::max(0, std::min(width - 1, x));
    int ny = std::max(0, std::min(height - 1, y));

    return data[(ny * width + nx) * channels + c];
  }

  const float &at(int c, int x, int y) const {
    // Check if channel is valid
    if (c < 0 || c >= channels) {
      throw std::out_of_range("Channel index out of range");
    }

    // Clamp x and y coordinates to valid image boundaries
    int nx = std::max(0, std::min(width - 1, x));
    int ny = std::max(0, std::min(height - 1, y));

    return data[(ny * width + nx) * channels + c];
  }

  // Set pixel value at (c, x, y)
  void set(int c, int x, int y, float value) { at(c, x, y) = value; }

  // Fill the entire image with a specific value
  void fill(float value) { std::fill(data.begin(), data.end(), value); }

  // Normalize image data to range 0-1
  void normalize() {
    if (data.empty())
      return;

    // Find min and max values across all channels
    float min_val = data[0];
    float max_val = data[0];

    for (const float &val : data) {
      min_val = std::min(min_val, val);
      max_val = std::max(max_val, val);
    }

    // If max equals min, avoid division by zero
    if (max_val == min_val) {
      // If all values are the same, set to 0 or 1 depending on value
      float normalized_val = (min_val > 0.5f) ? 1.0f : 0.0f;
      std::fill(data.begin(), data.end(), normalized_val);
      return;
    }

    // Normalize all values to 0-1 range
    float range = max_val - min_val;
    for (float &val : data) {
      val = (val - min_val) / range;
    }
  }

  int getWidth() const { return width; }
  int getHeight() const { return height; }
  int getChannels() const { return channels; }
  const std::vector<float> &getData() const { return data; }
};

#endif // IMAGE_HPP
