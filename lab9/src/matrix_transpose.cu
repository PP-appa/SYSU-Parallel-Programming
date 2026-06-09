#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << ": " << cudaGetErrorString(err) << std::endl;        \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

struct Options {
    int n = 1024;
    int block_x = 32;
    int block_y = 8;
    int repeats = 10;
    unsigned seed = 20260609;
    bool print_matrices = false;
    std::string kernel = "tiled";
};

__global__ void transpose_naive(const float *input, float *output, int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        output[col * n + row] = input[row * n + col];
    }
}

__global__ void transpose_tiled(const float *input, float *output, int n)
{
    extern __shared__ float tile[];
    int tile_dim = blockDim.x;
    int block_rows = blockDim.y;

    int x = blockIdx.x * tile_dim + threadIdx.x;
    int y = blockIdx.y * tile_dim + threadIdx.y;

    for (int j = 0; j < tile_dim; j += block_rows) {
        if (x < n && y + j < n) {
            tile[(threadIdx.y + j) * tile_dim + threadIdx.x] =
                input[(y + j) * n + x];
        }
    }
    __syncthreads();

    x = blockIdx.y * tile_dim + threadIdx.x;
    y = blockIdx.x * tile_dim + threadIdx.y;

    for (int j = 0; j < tile_dim; j += block_rows) {
        if (x < n && y + j < n) {
            output[(y + j) * n + x] =
                tile[threadIdx.x * tile_dim + threadIdx.y + j];
        }
    }
}

long parse_long(const char *text, const char *name)
{
    char *end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (*text == '\0' || *end != '\0') {
        std::cerr << name << " must be an integer, got " << text << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return value;
}

void print_usage(const char *program)
{
    std::cerr
        << "Usage: " << program << " [options]\n"
        << "  --n N                 square matrix size, 512 <= N <= 2048\n"
        << "  --kernel naive|tiled  transpose kernel variant\n"
        << "  --block-x X           CUDA block x dimension\n"
        << "  --block-y Y           CUDA block y dimension\n"
        << "  --repeats R           timed kernel repeats, default 10\n"
        << "  --seed S              random matrix seed\n"
        << "  --print               print A and A^T (use only for small N)\n";
}

Options parse_args(int argc, char **argv)
{
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                std::cerr << name << " requires a value" << std::endl;
                std::exit(EXIT_FAILURE);
            }
            return argv[++i];
        };

        if (arg == "--n") {
            opt.n = static_cast<int>(parse_long(require_value("--n"), "--n"));
        } else if (arg == "--kernel") {
            opt.kernel = require_value("--kernel");
        } else if (arg == "--block-x") {
            opt.block_x = static_cast<int>(
                parse_long(require_value("--block-x"), "--block-x"));
        } else if (arg == "--block-y") {
            opt.block_y = static_cast<int>(
                parse_long(require_value("--block-y"), "--block-y"));
        } else if (arg == "--repeats") {
            opt.repeats = static_cast<int>(
                parse_long(require_value("--repeats"), "--repeats"));
        } else if (arg == "--seed") {
            opt.seed = static_cast<unsigned>(
                parse_long(require_value("--seed"), "--seed"));
        } else if (arg == "--print") {
            opt.print_matrices = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }

    if (opt.n < 512 || opt.n > 2048) {
        std::cerr << "--n must be in [512, 2048]" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (opt.block_x <= 0 || opt.block_y <= 0 || opt.block_x * opt.block_y > 1024) {
        std::cerr << "block dimensions must be positive and contain at most 1024 threads"
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (opt.kernel == "tiled" &&
        (opt.block_y > opt.block_x || opt.block_x % opt.block_y != 0)) {
        std::cerr << "tiled kernel requires block_y <= block_x and block_x % block_y == 0"
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (opt.repeats <= 0) {
        std::cerr << "--repeats must be positive" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (opt.kernel != "naive" && opt.kernel != "tiled") {
        std::cerr << "--kernel must be either naive or tiled" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return opt;
}

void fill_matrix(std::vector<float> &matrix, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float &value : matrix) {
        value = dist(rng);
    }
}

float max_abs_error(const std::vector<float> &input,
                    const std::vector<float> &output,
                    int n)
{
    float error = 0.0f;
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            float expected = input[col * n + row];
            float got = output[row * n + col];
            error = std::max(error, std::fabs(expected - got));
        }
    }
    return error;
}

void print_matrix(const std::vector<float> &matrix, int n, const char *name)
{
    std::cout << name << " =\n";
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            std::cout << std::setw(8) << std::setprecision(4)
                      << matrix[row * n + col] << ' ';
        }
        std::cout << '\n';
    }
}

int main(int argc, char **argv)
{
    Options opt = parse_args(argc, argv);
    size_t element_count = static_cast<size_t>(opt.n) * opt.n;
    size_t bytes = element_count * sizeof(float);

    std::vector<float> host_input(element_count);
    std::vector<float> host_output(element_count, std::numeric_limits<float>::quiet_NaN());
    fill_matrix(host_input, opt.seed);

    float *device_input = nullptr;
    float *device_output = nullptr;
    CHECK_CUDA(cudaMalloc(&device_input, bytes));
    CHECK_CUDA(cudaMalloc(&device_output, bytes));
    CHECK_CUDA(cudaMemcpy(device_input, host_input.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(device_output, 0, bytes));

    dim3 block(opt.block_x, opt.block_y);
    dim3 grid;
    if (opt.kernel == "naive") {
        grid = dim3((opt.n + opt.block_x - 1) / opt.block_x,
                    (opt.n + opt.block_y - 1) / opt.block_y);
    } else {
        grid = dim3((opt.n + opt.block_x - 1) / opt.block_x,
                    (opt.n + opt.block_x - 1) / opt.block_x);
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int r = 0; r < opt.repeats; ++r) {
        if (opt.kernel == "naive") {
            transpose_naive<<<grid, block>>>(device_input, device_output, opt.n);
        } else {
            size_t shared_bytes = static_cast<size_t>(opt.block_x) * opt.block_x *
                                  sizeof(float);
            transpose_tiled<<<grid, block, shared_bytes>>>(device_input,
                                                           device_output,
                                                           opt.n);
        }
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    float time_ms = total_ms / opt.repeats;
    double bandwidth_gb_s = (2.0 * static_cast<double>(bytes)) / (time_ms * 1.0e6);

    CHECK_CUDA(cudaMemcpy(host_output.data(), device_output, bytes, cudaMemcpyDeviceToHost));
    float error = max_abs_error(host_input, host_output, opt.n);

    if (opt.print_matrices) {
        if (opt.n > 16) {
            std::cerr << "--print is intended for N <= 16; suppressing matrix dump"
                      << std::endl;
        } else {
            print_matrix(host_input, opt.n, "A");
            print_matrix(host_output, opt.n, "AT");
        }
    }

    std::cout << std::fixed << std::setprecision(6)
              << "matrix_size=" << opt.n
              << " kernel=" << opt.kernel
              << " block_x=" << opt.block_x
              << " block_y=" << opt.block_y
              << " repeats=" << opt.repeats
              << " time_ms=" << time_ms
              << " bandwidth_gb_s=" << bandwidth_gb_s
              << " max_abs_error=" << error << std::endl;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(device_input));
    CHECK_CUDA(cudaFree(device_output));
    return error == 0.0f ? EXIT_SUCCESS : EXIT_FAILURE;
}
