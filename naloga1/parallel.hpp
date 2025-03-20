#ifndef PARALLEL_HPP
#define PARALLEL_HPP
#include "Image.hpp"
#include <omp.h>

inline Image calc_energy_par(Image in) {
  Image out(in.getWidth(), in.getHeight(), 1);
  
  #pragma omp parallel for collapse(2) schedule(dynamic)
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

inline Image id_seams_par(Image energy) {
  const thr = omp_get_max_threads()
  Image out(in.getWidth() - 1, in.getHeight(), in.getChannels());


  for(int t=0; t < thr; t++){
    // each thread should compute the triangle
    const triangle_width = out.getWidth() / 
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

inline Image rem_seam_par(Image in, Image seams) {

}
#endif // PARALLEL_HPP