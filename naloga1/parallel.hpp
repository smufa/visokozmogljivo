#ifndef PARALLEL_HPP
#define PARALLEL_HPP
#include "Image.hpp"
#include <omp.h>

inline Image calc_energy_par(Image in)
{
  Image out(in.getWidth(), in.getHeight(), 1);

#pragma omp parallel for collapse(2) schedule(dynamic)
  for (int x = 0; x < in.getWidth(); x++)
  {
    for (int y = 0; y < in.getHeight(); y++)
    {
      std::vector<float> energies = {0, 0, 0};
      for (int c = 0; c < in.getChannels(); c++)
      {
        energies[c] = std::sqrt(
            std::pow(-in.at(c, x - 1, y - 1) - 2 * in.at(c, x, y - 1) -
                         in.at(c, x + 1, y - 1) + in.at(c, x - 1, y + 1) +
                         2 * in.at(c, x, y + 1) + in.at(c, x + 1, y + 1),
                     2) +
            std::pow(+in.at(c, x - 1, y - 1) + 2 * in.at(c, x - 1, y) +
                         in.at(c, x - 1, y + 1) - in.at(c, x + 1, y - 1) -
                         2 * in.at(c, x + 1, y) - in.at(c, x + 1, y + 1),
                     2));
      }
      out.set(0, x, y, (energies[0] + energies[1] + energies[2]) / 3);
    }
  }
  return out;
}

inline Image id_seams_par(Image energy)
{
  Image out(energy.getWidth(), energy.getHeight(), energy.getChannels());
  const int num_triangles = 4;
  const int strip_size = (out.getWidth() / num_triangles) / 2; // half of full traingle base
  const int strips = energy.getHeight() / strip_size;
  printf("strip size: %d, strips: %d\n", strip_size, strips);

  // divide the image in (NON)equal horizontal strips
  for (int strip = 0; strip * strip_size < out.getHeight(); strip++)
  {
    // calculate y boundaries for strips processing currently
    const int strip_from = strip * strip_size;
    const int strip_to = std::min(strip_from + strip_size, out.getHeight());
    printf("strip from: %d, to: %d \n", strip_from, strip_to);

    for (int triangle_index = 0; triangle_index < num_triangles; triangle_index++)
    {
      // each thread should compute the triangle
      // compute the triangle untill the width at the top is 0
      int max_triangle_width = out.getWidth() / num_triangles;
      int triangle_width = out.getWidth() / num_triangles;
      int level = strip_from;

      while (triangle_width > 0) // decrease triangle width by 2 every time you go up in Y
      {
        // from / to in x axis
        const int from = (max_triangle_width * triangle_index + level - strip_from);
        const int to = std::min(from + triangle_width, out.getWidth());
        for (int i = from; i < to; i++)
        {
          const int prev_layer = out.getHeight() - level - 1;
          const float min_previous_layer = std::min(
              std::min(
                  energy.at(0, i - 1, prev_layer),
                  energy.at(0, i, prev_layer)),
              energy.at(0, i + 1, prev_layer));

          out.set(0, i, prev_layer + 1, 1.0);
          // out.set(0, i, prev_layer + 1, min_previous_layer + energy.at(0, i, prev_layer + 1));
        }
        level += 1;
        triangle_width -= 2;
      }
    }
    // up facing triangles are now computed

    // barier ? all threads have computed all the triangles in a strip

    // compute down facing triangles
    for (int triangle_index = 0; triangle_index <= num_triangles; triangle_index++)
    {
      // each thread should compute the triangle
      // compute the triangle untill the width at the top is 0
      int triangle_width = 2;
      int global_level = strip_from + 1; // since the bottom layer was computed by up-facing triangle; +1 is the new level we compute
      const int max_triangle_width = out.getWidth() / num_triangles;

      while (triangle_width < max_triangle_width) // increase the triangle size by 2, building upwards.
      {
        // from / to in x axis
        int from = triangle_index * max_triangle_width - (global_level - strip_from); // calculate the from index (negative for first triangle)
        const int to = std::min(from + triangle_width, out.getWidth());               // calculate real to
        from = std::max(from, 0);                                                     // correct the from, to work on first triangle too

        for (int i = from; i < to; i++)
        {
          const int prev_layer = out.getHeight() - global_level - 1;
          const float min_previous_layer = std::min(
              std::min(
                  energy.at(0, i - 1, prev_layer),
                  energy.at(0, i, prev_layer)),
              energy.at(0, i + 1, prev_layer));

          // out.set(0, i, prev_layer + 1, min_previous_layer + energy.at(0, i, prev_layer + 1));
          out.set(0, i, prev_layer + 1, 0.5);
        }
        global_level += 1;
        triangle_width += 2;
      }
    }
  }

  //   float seam_score = std::numeric_limits<float>::infinity();
  //   for(int x = 0; x < in.getWidth(); x++) {
  //       if(seams.at(0, x, in.getHeight() - 1) < seam_score) {
  //           seam_score = seams.at(0, x, in.getHeight() - 1);
  //           seam_index = x;
  //       }
  //   }

  //   for(int y = in.getHeight() - 2; y > 0; y--) {
  //       bool skipped = false;
  //       for(int x = 0; x < out.getWidth(); x++) {
  //           if(!skipped && x == seam_index) {
  //               if(seams.at(0, x-1, y-1) < seams.at(0, x, y-1) && seams.at(0, x-1, y-1) < seams.at(0, x+1, y-1)) {
  //                   seam_index = x - 1;
  //               } else if (seams.at(0, x, y-1) < seams.at(0, x+1, y-1)) {
  //                   seam_index = x;
  //               } else {
  //                   seam_index = x + 1;
  //               }
  //               skipped = true;
  //           } else {
  //               for(int c = 0; c < out.getChannels(); c++) {
  //                   out.set(c, skipped ? x-1 : x, y, in.at(c, x, y));
  //               }
  //           }
  //       }
  //   }
  return out;
}

inline Image rem_seam_par(Image in, Image seams)
{
  return in;
}
#endif // PARALLEL_HPP