#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <string>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << ": " << cudaGetErrorString(err) << std::endl;        \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

__global__ void hello_kernel()
{
    printf("Hello World from Thread (%d, %d) in Block %d!\n",
           threadIdx.x,
           threadIdx.y,
           blockIdx.x);
}

int parse_bounded_int(const char *text, const char *name)
{
    char *end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (*text == '\0' || *end != '\0' || value < 1 || value > 32) {
        std::cerr << name << " must be an integer in [1, 32], got " << text
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <n_blocks> <block_dim_x> <block_dim_y>\n"
                  << "Each argument must be in [1, 32]." << std::endl;
        return EXIT_FAILURE;
    }

    int n = parse_bounded_int(argv[1], "n_blocks");
    int m = parse_bounded_int(argv[2], "block_dim_x");
    int k = parse_bounded_int(argv[3], "block_dim_y");

    std::cout << "Hello World from the host!" << std::endl;
    hello_kernel<<<n, dim3(m, k)>>>();
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    return EXIT_SUCCESS;
}
