#ifndef SEQUENTIAL_HPP
#define SEQUENTIAL_HPP
#include "Image.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

inline Image calc_energy_seq(Image in) {
  Image out(in.getWidth(), in.getHeight(), 1);
  for (int x = 0; x < in.getWidth(); x++) {
    for (int y = 0; y < in.getHeight(); y++) {
      std::vector<float> energies = {0, 0, 0};
      for (int c = 0; c < in.getChannels(); c++) {
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

inline Image id_seams_seq(Image energy) {
  Image out = energy;
  int height = out.getHeight();
  for (int y = height - 2; y >= 0; y--) {
    for (int x = 0; x < out.getWidth(); x++) {
      out.set(0, x, y,
              out.at(0, x, y) +
                  std::min({out.at(0, x - 1, y + 1), out.at(0, x, y + 1),
                            out.at(0, x + 1, y + 1)}));
    }
  }
  return out;
}

inline Image rem_seam_seq(Image in, Image seams) {
  Image out(in.getWidth() - 1, in.getHeight(), in.getChannels());
  std::vector<int> seam(in.getHeight());

  // Find start of seam
  float seam_score = std::numeric_limits<float>::infinity();
  int seam_start;
  for (int x = 0; x < in.getWidth(); x++) {
    if (seams.at(0, x, 0) < seam_score) {
      seam_score = seams.at(0, x, 0);
      seam[0] = x;
    }
  }

  // Find entire seam
  for (int y = 1; y < in.getHeight(); y++) {
    if (seams.at(0, seam[y - 1], y) < seams.at(0, seam[y - 1] - 1, y) &&
        seams.at(0, seam[y - 1], y) < seams.at(0, seam[y - 1] + 1, y)) {
      seam[y] = seam[y - 1];
    } else if (seams.at(0, seam[y - 1] - 1, y) <
               seams.at(0, seam[y - 1] + 1, y)) {
      seam[y] = seam[y - 1] - 1;
    } else {
      seam[y] = seam[y - 1] + 1;
    }
  }

  for (int y = 0; y < in.getHeight(); y++) {
    bool skipped = false;
    for (int x = 0; x < in.getWidth(); x++) {
      if (x == seam[y]) {
        skipped = true;
      } else {
        for (int c = 0; c < out.getChannels(); c++) {
          out.set(c, skipped ? x - 1 : x, y, in.at(c, x, y));
        }
      }
    }
  }
  return out;
}

#endif // SEQUENTIAL_HPP