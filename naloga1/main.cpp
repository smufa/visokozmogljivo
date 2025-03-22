#include "Image.hpp"
#include "parallel.hpp"
#include "sequential.hpp"
#include <chrono>
#include <iostream>

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

// Carve function now uses the Image class
Image carve(const Image &img, int pixels_to_remove) {
  auto current = img;
  std::chrono::time_point<std::chrono::system_clock> t1, t2, t3, t4, t5;
  t1 = high_resolution_clock::now();
  for (int i = 0; i < pixels_to_remove; i++) {
    t2 = high_resolution_clock::now();
    auto energy = calc_energy_par(current);
    t3 = high_resolution_clock::now();

    auto seams = id_seams_par(energy);
    t4 = high_resolution_clock::now();

    current = rem_seam_par(current, seams);
    t5 = high_resolution_clock::now();
    seams.normalize();
    seams.save("testout.png");
  }
  t5 = high_resolution_clock::now();

  auto ms_int_ene = duration_cast<milliseconds>(t3 - t2);
  auto ms_int_seams = duration_cast<milliseconds>(t4 - t3);
  auto ms_int_rem = duration_cast<milliseconds>(t5 - t4);
  std::cout << "Energy: " << ms_int_ene.count() << "ms\n";
  std::cout << "Seam process: " << ms_int_seams.count() << "ms\n";
  std::cout << "Seam energy: " << ms_int_rem.count() << "ms\n";

  auto ms_int = duration_cast<milliseconds>(t5 - t1);
  std::cout << "Cumulative: " << ms_int.count() << "ms\n";
  return current;
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " <input> <output> <pixels_to_remove>\n";
    return EXIT_FAILURE;
  }

  // Process parameters
  std::string input_filename = argv[1];
  std::string output_filename = argv[2];
  int pixels_to_remove;

  try {
    pixels_to_remove = std::stoi(argv[3]);
  } catch (...) {
    std::cerr << "Invalid pixel count\n";
    return EXIT_FAILURE;
  }

  try {
    // Load image
    Image input = Image::load(input_filename);

    if (pixels_to_remove <= 0 || pixels_to_remove >= input.getWidth()) {
      std::cerr << "Invalid removal value (must be 1-" << input.getWidth() - 1
                << ")\n";
      return EXIT_FAILURE;
    }

    // Process image
    Image output = carve(input, pixels_to_remove);

    // Save result
    if (!output.save(output_filename)) {
      std::cerr << "Error saving image\n";
      return EXIT_FAILURE;
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
