#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <vector>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <stdexcept>
#include <iostream>

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
    static std::string get_extension(const std::string& filename) {
        size_t dot = filename.find_last_of(".");
        if (dot == std::string::npos) return "";
        std::string ext = filename.substr(dot + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        return ext;
    }

public:
    // Constructor for empty image
    Image() : width(0), height(0), channels(0) {}

    // Constructor with data
    Image(const std::vector<float>& img_data, int w, int h, int c)
        : data(img_data), width(w), height(h), channels(c) {}
    
    // Constructor for creating an image of specified size with zeros
    Image(int w, int h, int c) : width(w), height(h), channels(c) {
        data.resize(w * h * c, 0.0f);
    }

    // Load image from file
    static Image load(const std::string& filename) {
        int width, height, channels;
        unsigned char* img_data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
        
        if (!img_data) {
            throw std::runtime_error(std::string("Error loading image: ") + stbi_failure_reason());
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
    bool save(const std::string& filename) const {
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
        }
        else if (ext == "jpg" || ext == "jpeg") {
            success = stbi_write_jpg(filename.c_str(), width, height, channels, 
                                    temp_data.data(), 90);
        }
        else {
            std::cerr << "Unsupported format. Use .png or .jpg" << std::endl;
            return false;
        }

        return success != 0;
    }

    // Indexing function (channel, x, y)
    float& at(int c, int x, int y) {
        if (c < 0 || c >= channels || x < 0 || x >= width || y < 0 || y >= height) {
            throw std::out_of_range("Image index out of range");
        }
        return data[(y * width + x) * channels + c];
    }

    const float& at(int c, int x, int y) const {
        if (c < 0 || c >= channels || x < 0 || x >= width || y < 0 || y >= height) {
            throw std::out_of_range("Image index out of range");
        }
        return data[(y * width + x) * channels + c];
    }

    // Set pixel value at (c, x, y)
    void set(int c, int x, int y, float value) {
        at(c, x, y) = value;
    }

    // Fill the entire image with a specific value
    void fill(float value) {
        std::fill(data.begin(), data.end(), value);
    }

    int getWidth() const { return width; }
    int getHeight() const { return height; }
    int getChannels() const { return channels; }
    const std::vector<float>& getData() const { return data; }
};

#endif // IMAGE_HPP
