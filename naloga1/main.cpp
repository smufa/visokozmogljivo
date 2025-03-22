#include <iostream>
#include "Image.hpp"
#include "sequential.hpp"
#include "parallel.hpp"

// Carve function now uses the Image class
Image carve(const Image &img, int pixels_to_remove)
{
    auto current = img;
    for (int i = 0; i < pixels_to_remove; i++)
    {
        auto energy = calc_energy_seq(current);
        auto seams = id_seams_par(energy);
        seams.normalize();
        seams.save("testout.png");

        // auto seams2 = id_seams_seq(energy);
        // seams2.normalize();
        // seams2.save("testout2.png");
        current = rem_seam_seq(current, seams);
    }
    return current;
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <input> <output> <pixels_to_remove>\n";
        return EXIT_FAILURE;
    }

    // Process parameters
    std::string input_filename = argv[1];
    std::string output_filename = argv[2];
    int pixels_to_remove;

    try
    {
        pixels_to_remove = std::stoi(argv[3]);
    }
    catch (...)
    {
        std::cerr << "Invalid pixel count\n";
        return EXIT_FAILURE;
    }

    try
    {
        // Load image
        Image input = Image::load(input_filename);

        if (pixels_to_remove <= 0 || pixels_to_remove >= input.getWidth())
        {
            std::cerr << "Invalid removal value (must be 1-" << input.getWidth() - 1 << ")\n";
            return EXIT_FAILURE;
        }

        // Process image
        Image output = carve(input, pixels_to_remove);

        // Save result
        if (!output.save(output_filename))
        {
            std::cerr << "Error saving image\n";
            return EXIT_FAILURE;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
