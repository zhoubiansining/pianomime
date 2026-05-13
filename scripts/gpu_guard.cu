#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <unistd.h>

static void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
    std::exit(1);
  }
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "Usage: %s GPU_ID GUARD_GB STOP_FILE\n", argv[0]);
    return 2;
  }

  int gpu = std::atoi(argv[1]);
  double gb = std::atof(argv[2]);
  std::string stop_file = argv[3];
  size_t bytes = static_cast<size_t>(gb * 1024.0 * 1024.0 * 1024.0);

  check(cudaSetDevice(gpu), "cudaSetDevice");
  void* ptr = nullptr;
  check(cudaMalloc(&ptr, bytes), "cudaMalloc");
  check(cudaMemset(ptr, 1, 1), "cudaMemset");
  check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  std::printf("guarding GPU %d with %.2f GiB until %s exists\n",
              gpu, gb, stop_file.c_str());
  std::fflush(stdout);

  while (access(stop_file.c_str(), F_OK) != 0) {
    std::this_thread::sleep_for(std::chrono::seconds(30));
  }

  cudaFree(ptr);
  std::printf("released GPU %d\n", gpu);
  return 0;
}
