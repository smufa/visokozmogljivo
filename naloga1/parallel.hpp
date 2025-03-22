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
  Image out = energy;
  const int num_triangles = 20;
  const int strip_size = std::ceil(((float)out.getWidth() / (float)num_triangles) / 2.0); // half of full traingle base
  printf("num: %d, wi: %d", num_triangles, out.getWidth());

  const int strips = out.getHeight() / strip_size;
  printf("strip size: %d, strips: %d\n", strip_size, strips);

  // divide the image in (NON)equal horizontal strips
  // for (int strip = 0; strip * strip_size < out.getHeight(); strip++)
  {
    for (int strip = 0; strip < strips; strip++)
    {
      // calculate y boundaries for strips processing currently
      const int strip_from = strip * strip_size + 1;
      // const int strip_to = std::min(strip_from + strip_size - 1, out.getHeight());
      printf("strip from: %d,  \n", strip_from);
#pragma omp parallel for
      for (int triangle_index = 0; triangle_index <= num_triangles; triangle_index++)
      {
        // each thread should compute the triangle
        // compute the triangle untill the width at the top is 0
        int max_triangle_width = out.getWidth() / num_triangles;
        int triangle_width = out.getWidth() / num_triangles;
        int level = strip_from + 1;

        while (triangle_width > 0) // decrease triangle width by 2 every time you go up in Y
        {
          // from / to in x axis
          const int from = (max_triangle_width * triangle_index + level - strip_from) - 1;
          const int to = std::min(from + triangle_width, out.getWidth());
          const int write_layer = out.getHeight() - level;
          const int read_layer = write_layer + 1;
          if (write_layer < 0 || read_layer < 0)
          {
            break;
          }
          // printf("prev: %d \n", prev_layer);
          for (int i = from; i < to; i++)
          {
            const float min_previous_layer = std::min(
                std::min(
                    out.at(0, i - 1, read_layer),
                    out.at(0, i, read_layer)),
                out.at(0, i + 1, read_layer));

            const float cost = (min_previous_layer + out.at(0, i, write_layer));

            // printf("cost: %f", cost);
            out.set(0, i, write_layer, cost);
            // out.set(0, i, read_layer, 0.5);
            // out.set(0, i, write_layer, 1);
          }
          level += 1;
          triangle_width -= 2;
          // printf("width: %d", triangle_width);
        }
      }
// up facing triangles are now computed

// barier -  all threads have computed all the triangles in a strip
#pragma omp barrier

#pragma omp parallel for
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

          // printf("prev_layer: %d \n", prev_layer);
          const int write_layer = out.getHeight() - global_level - 1;
          const int read_layer = write_layer + 1;
          if (write_layer < 0 || read_layer < 0)
          {
            break;
          }

          for (int i = from; i < to; i++)
          {
            const float min_previous_layer = std::min(
                std::min(
                    out.at(0, i - 1, read_layer),
                    out.at(0, i, read_layer)),
                out.at(0, i + 1, read_layer));

            out.set(0, i, write_layer, min_previous_layer + out.at(0, i, write_layer));
            // out.set(0, i, write_layer, 0.5);
            // out.set(0, i, read_layer, 0.4);
          }
          global_level += 1;
          triangle_width += 2;
        }
      }
    }
  }

  return out;
}

inline Image rem_seam_par(Image in, Image seams)
{
  return in;
}
#endif // PARALLEL_HPP