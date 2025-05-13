#include "../helpers.hpp"
#include <iostream>
#include <ostream>
#include <string>
#include <chrono>

int main(int argc, char *argv[]) {
  std::cout << "Gray-Scott CPU implementation" << std::endl;

  double diffusion_rate_a = 0.0;
  double diffusion_rate_b = 0.0;
  double feed_rate = 0.0;
  double kill_rate = 0.0;
  int time_steps = 0;
  double delta_t = 0.0;
  std::string image_file = "";
  std::string output_file = "output.png";

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

      // Make a copy of the image
      size_t image_size = width * height * channels;
      float *image_copy = new float[image_size];
      std::copy(img, img + image_size, image_copy);

      // Main loop that goes through the image twice
      auto start_time = std::chrono::high_resolution_clock::now();
      for (int step = 0; step < time_steps; ++step) {
        for (int i = 0; i < height; ++i) {
          for (int j = 0; j < width; ++j) {
            // Access pixel data here
            float U = imageAt(img, width, height, j, i, 0);
            float V = imageAt(img, width, height, j, i, 1);

            // Calculate Laplacian
            float laplacian_U = imageAt(img, width, height, j + 1, i, 0) +
                                imageAt(img, width, height, j - 1, i, 0) +
                                imageAt(img, width, height, j, i + 1, 0) +
                                imageAt(img, width, height, j, i - 1, 0) -
                                4.0f * U;

            float laplacian_V = imageAt(img, width, height, j + 1, i, 1) +
                                imageAt(img, width, height, j - 1, i, 1) +
                                imageAt(img, width, height, j, i + 1, 1) +
                                imageAt(img, width, height, j, i - 1, 1) -
                                4.0f * V;

            float UV2 = U * V * V;

            imageAt(image_copy, width, height, j, i, 0) =
                U + delta_t * (diffusion_rate_a * laplacian_U - UV2 +
                               feed_rate * (1 - U));
            imageAt(image_copy, width, height, j, i, 1) =
                V + delta_t * (diffusion_rate_b * laplacian_V + UV2 -
                               (feed_rate + kill_rate) * V);
          }
        }
        std::swap(img, image_copy);
      }
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
      std::cout << "Main loop execution time: " << duration.count() << " ms" << std::endl;

      writeImage(output_file.c_str(), width, height, channels, img);
      
      delete[] image_copy;
      delete[] img;
    }
  }

  return 0;
}