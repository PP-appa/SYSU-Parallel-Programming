# Lab 3 - Parallel Matrix Multiplication and Array Summation via Pthreads

This directory contains the implementations for Lab 3 of the Parallel Programming and Algorithms course. The lab primarily uses the POSIX Threads (Pthreads) standard to implement parallel execution on multi-core architectures. 

## Structure

- **`src/pthread_gemm.cpp`**: Multi-threaded Matrix Multiplication with block partition execution.
- **`src/pthread_sum.cpp`**: Parallel Array Summation optimized with thread-local registers to avoid False Sharing.
- **`src/pthread_sum_falsesharing.cpp`**: Educational implementation causing cache line invalidation intentionally (False Sharing) to benchmark its performance penalties.
- **`report.md`**: Experimental observations tracking the execution time, speedup rates, and deep architectural analyses.

## Compilation

A basic C++ environment supporting `-lpthread` is required (e.g. GCC on Linux/WSL). Run the following commands from the root of `lab3`:

```bash
mkdir -p bin

# 1. Compile Parallel Matrix Multiplication
g++ -O3 src/pthread_gemm.cpp -o bin/pthread_gemm -lpthread

# 2. Compile Parallel Array Summation
g++ -O3 src/pthread_sum.cpp -o bin/pthread_sum -lpthread

# 3. Compile False Sharing benchmark 
# (Note: Use -O0 to explicitly prevent GCC from optimizing away the False Sharing behavior)
g++ -O0 src/pthread_sum_falsesharing.cpp -o bin/pthread_sum_falsesharing_O0 -lpthread
g++ -O0 src/pthread_sum.cpp -o bin/pthread_sum_O0 -lpthread
```

## Running the Examples

Matrix Multiplication requires four arguments: `<m> <n> <k> <num_threads>`.
```bash
./bin/pthread_gemm 2048 2048 2048 16
```

Array summation requires two arguments: `<array_size_n> <num_threads>`.
```bash
./bin/pthread_sum 128000000 16
./bin/pthread_sum_O0 128000000 16
./bin/pthread_sum_falsesharing_O0 128000000 16
```

You can also find automated shell scripts like `run_experiments.sh` to populate multiple test vectors sequentially.
