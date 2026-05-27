#include <mpi.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

static double lcg(double& seed) {
  static constexpr double A = 16807.0;
  static constexpr double M = 2147483647.0;
  const double k = std::floor(seed / 127773.0);
  seed = A * (seed - k * 127773.0) - k * 2836.0;
  if (seed < 0.0) seed += M;
  return seed / M;
}

static bool is_power_of_two(int x) { return x > 0 && (x & (x - 1)) == 0; }

static int bit_reverse(int x, int bits) {
  int r = 0;
  for (int i = 0; i < bits; ++i) {
    r = (r << 1) | (x & 1);
    x >>= 1;
  }
  return r;
}

static void fft_mpi_1d(std::vector<std::complex<double>>& local,
                       std::vector<std::complex<double>>& global,
                       int n,
                       int rank,
                       int size,
                       int sign) {
  const int local_n = n / size;
  const int bits = static_cast<int>(std::log2(n));

  MPI_Allgather(local.data(), local_n, MPI_CXX_DOUBLE_COMPLEX, global.data(), local_n,
                MPI_CXX_DOUBLE_COMPLEX, MPI_COMM_WORLD);
  for (int loc = 0; loc < local_n; ++loc) {
    const int idx = rank * local_n + loc;
    local[loc] = global[bit_reverse(idx, bits)];
  }

  for (int len = 2; len <= n; len <<= 1) {
    const int half = len >> 1;
    MPI_Allgather(local.data(), local_n, MPI_CXX_DOUBLE_COMPLEX, global.data(), local_n,
                  MPI_CXX_DOUBLE_COMPLEX, MPI_COMM_WORLD);

    std::vector<std::complex<double>> next(local_n);
    for (int loc = 0; loc < local_n; ++loc) {
      const int idx = rank * local_n + loc;
      const int block = (idx / len) * len;
      const int j = idx - block;
      if (j < half) {
        const int i1 = idx;
        const int i2 = idx + half;
        const double ang = sign * -2.0 * M_PI * static_cast<double>(j) / static_cast<double>(len);
        const std::complex<double> w = std::polar(1.0, ang);
        next[loc] = global[i1] + w * global[i2];
      } else {
        const int jp = j - half;
        const int i1 = idx - half;
        const int i2 = idx;
        const double ang = sign * -2.0 * M_PI * static_cast<double>(jp) / static_cast<double>(len);
        const std::complex<double> w = std::polar(1.0, ang);
        next[loc] = global[i1] - w * global[i2];
      }
    }
    local.swap(next);
  }
  MPI_Allgather(local.data(), local_n, MPI_CXX_DOUBLE_COMPLEX, global.data(), local_n,
                MPI_CXX_DOUBLE_COMPLEX, MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = 1 << 12;
  int nits = 20;
  if (argc >= 2) n = std::atoi(argv[1]);
  if (argc >= 3) nits = std::atoi(argv[2]);

  if (!is_power_of_two(n)) {
    if (rank == 0) std::cerr << "N must be a power of two.\n";
    MPI_Finalize();
    return 1;
  }
  if (n % size != 0) {
    if (rank == 0) std::cerr << "N must be divisible by process count.\n";
    MPI_Finalize();
    return 1;
  }

  const int local_n = n / size;
  std::vector<std::complex<double>> full(n), original(n), global(n), local(local_n);

  if (rank == 0) {
    double seed = 331.0;
    for (int i = 0; i < n; ++i) {
      full[i] = {lcg(seed), lcg(seed)};
      original[i] = full[i];
    }
  }

  MPI_Scatter(full.data(), local_n, MPI_CXX_DOUBLE_COMPLEX, local.data(), local_n,
              MPI_CXX_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

  // Correctness check: forward then inverse FFT should recover original values.
  fft_mpi_1d(local, global, n, rank, size, +1);
  local.assign(global.begin() + rank * local_n, global.begin() + (rank + 1) * local_n);
  fft_mpi_1d(local, global, n, rank, size, -1);

  double local_err = 0.0;
  if (rank == 0) {
    for (int i = 0; i < n; ++i) {
      const auto recovered = global[i] / static_cast<double>(n);
      const auto diff = recovered - original[i];
      local_err += std::norm(diff);
    }
  }

  double err = 0.0;
  MPI_Bcast(&local_err, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  err = std::sqrt(local_err / static_cast<double>(n));

  // Benchmark on deterministic zero input like the serial sample's timed branch.
  std::fill(local.begin(), local.end(), std::complex<double>(0.0, 0.0));
  MPI_Barrier(MPI_COMM_WORLD);
  const double t1 = MPI_Wtime();
  for (int it = 0; it < nits; ++it) {
    fft_mpi_1d(local, global, n, rank, size, +1);
    local.assign(global.begin() + rank * local_n, global.begin() + (rank + 1) * local_n);
    fft_mpi_1d(local, global, n, rank, size, -1);
    local.assign(global.begin() + rank * local_n, global.begin() + (rank + 1) * local_n);
  }
  const double t2 = MPI_Wtime();

  double elapsed = t2 - t1;
  double max_elapsed = 0.0;
  MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    const double flops = 2.0 * static_cast<double>(nits) * (5.0 * static_cast<double>(n) * std::log2(n));
    const double mflops = flops / 1.0e6 / max_elapsed;
    std::cout << "FFT_MPI\n";
    std::cout << "  procs=" << size << " N=" << n << " nits=" << nits << "\n";
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "  error=" << err << "\n";
    std::cout << "  total_time=" << max_elapsed << " s\n";
    std::cout << "  time_per_call=" << (max_elapsed / (2.0 * nits)) << " s\n";
    std::cout << "  mflops=" << mflops << "\n";
  }

  MPI_Finalize();
  return 0;
}
