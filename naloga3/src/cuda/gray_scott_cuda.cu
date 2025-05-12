#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  std::cout << "Gray-Scott CUDA implementation" << std::endl;

  double diffusion_rate_a = 0.0;
  double diffusion_rate_b = 0.0;
  double feed_rate = 0.0;
  double kill_rate = 0.0;
  int time_steps = 0;
  double delta_t = 0.0;
  std::string image_file = "";

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
    }
  }

  std::cout << "diffusion_rate_a: " << diffusion_rate_a << std::endl;
  std::cout << "diffusion_rate_b: " << diffusion_rate_b << std::endl;
  std::cout << "feed_rate: " << feed_rate << std::endl;
  std::cout << "kill_rate: " << kill_rate << std::endl;
  std::cout << "time_steps: " << time_steps << std::endl;
  std::cout << "delta_t: " << delta_t << std::endl;
  std::cout << "image_file: " << image_file << std::endl;

  return 0;
}