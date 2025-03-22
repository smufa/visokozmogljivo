#ifndef SEQUENTIAL_HPP
#define SEQUENTIAL_HPP
#include "Image.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

inline Image calc_energy_seq(Image in)
{
    Image out(in.getWidth(), in.getHeight(), 1);
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

// inline Image id_seams_seq(Image energy) {
//   Image out = energy;
//   for (int y = 1; y < out.getHeight(); y++) {
//     for (int x = 0; x < out.getWidth(); x++) {
//       out.set(0, x, y,
//               out.at(0, x, y) +
//                   std::min({out.at(0, x - 1, y - 1), out.at(0, x, y - 1),
//                             out.at(0, x + 1, y - 1)}));
//     }
//   }
//   return out;
// }

inline Image id_seams_seq(Image energy)
{
    Image out = energy;
    int height = out.getHeight();
    for (int y = height - 2; y >= 0; y--)
    {
        for (int x = 0; x < out.getWidth(); x++)
        {
            out.set(0, x, y,
                    out.at(0, x, y) +
                        std::min({out.at(0, x - 1, y + 1), out.at(0, x, y + 1),
                                  out.at(0, x + 1, y + 1)}));
        }
    }
    return out;
}

inline Image rem_seam_seq(Image in, Image seams)
{
    Image out(in.getWidth() - 1, in.getHeight(), in.getChannels());
    float seam_score = std::numeric_limits<float>::infinity();
    int seam_index;
    for (int x = 0; x < in.getWidth(); x++)
    {
        if (seams.at(0, x, in.getHeight() - 1) < seam_score)
        {
            seam_score = seams.at(0, x, in.getHeight() - 1);
            seam_index = x;
        }
    }
    for (int y = in.getHeight() - 2; y > 0; y--)
    {
        bool skipped = false;
        for (int x = 0; x < out.getWidth(); x++)
        {
            if (!skipped && x == seam_index)
            {
                if (seams.at(0, x - 1, y - 1) < seams.at(0, x, y - 1) && seams.at(0, x - 1, y - 1) < seams.at(0, x + 1, y - 1))
                {
                    seam_index = x - 1;
                }
                else if (seams.at(0, x, y - 1) < seams.at(0, x + 1, y - 1))
                {
                    seam_index = x;
                }
                else
                {
                    seam_index = x + 1;
                }
                skipped = true;
            }
            else
            {
                for (int c = 0; c < out.getChannels(); c++)
                {
                    out.set(c, skipped ? x - 1 : x, y, in.at(c, x, y));
                }
            }
        }
    }
    return out;
}

#endif // SEQUENTIAL_HPP