/**
 * @file mpi_gemm.cpp
 * @brief Parallel Matrix Multiplication using MPI Point-to-Point Communication (Lab 1)
 * @details Implements a typical Master-Worker pattern. Matrix data is simulated
 * using 1D std::vector (Row-major layout) for continuous MPI transmission.
 */

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

int main(int argc, char** argv) {
    // 1. Initialize MPI environment
    MPI_Init(&argc, &argv);

    int rank, size;
    // Get the rank of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    // Get the total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    // 2. Parse matrix dimensions (default m=128, n=128, k=128)
    int m = 128, n = 128, k = 128;
    if (argc >= 4) {
        m = std::atoi(argv[1]);
        n = std::atoi(argv[2]);
        k = std::atoi(argv[3]);
    }

    if (rank == 0) {
        // ====================================================================
        // Master Process (Rank 0): Initializes matrices, distributes tasks, 
        // and collects/assembles the final result.
        // ====================================================================
        
        std::vector<double> A(m * n);
        std::vector<double> B(n * k);
        std::vector<double> C(m * k, 0.0);

        // Initialize matrices A and B with random values
        std::srand(static_cast<unsigned>(std::time(nullptr)));
        for (int i = 0; i < m * n; ++i) A[i] = (double)rand() / RAND_MAX;
        for (int i = 0; i < n * k; ++i) B[i] = (double)rand() / RAND_MAX;

        double start_time = MPI_Wtime();

        if (size > 1) {
            // -- Phase 1: Broadcast entire matrix B to all workers --
            for (int dest = 1; dest < size; ++dest) {
                MPI_Send(B.data(), n * k, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            }

            // -- Phase 2: Partition matrix A by rows and distribute to all workers --
            int rows_per_worker = m / (size - 1);
            for (int dest = 1; dest < size; ++dest) {
                int start_row = (dest - 1) * rows_per_worker;
                // Assign any remaining rows to the last worker
                int num_rows = (dest == size - 1) ? (m - start_row) : rows_per_worker; 
                MPI_Send(A.data() + start_row * n, num_rows * n, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            }

            // -- Phase 3: Receive local results from workers and assemble into global matrix C --
            for (int src = 1; src < size; ++src) {
                int start_row = (src - 1) * rows_per_worker;
                int num_rows = (src == size - 1) ? (m - start_row) : rows_per_worker;
                // Receive directly into the target memory offset to avoid extra copying
                MPI_Recv(C.data() + start_row * k, num_rows * k, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

        } else {
            // When size == 1, Master executes the serial computation itself
            // Loop order i-k-j is used to optimize Cache hit rates
            for (int i = 0; i < m; ++i) {
                for (int p = 0; p < n; ++p) {
                    double r = A[i * n + p];
                    for (int j = 0; j < k; ++j) {
                        C[i * k + j] += r * B[p * k + j];
                    }
                }
            }
        }

        double end_time = MPI_Wtime();
        std::cout << "Time: " << end_time - start_time << " seconds" << std::endl;
        
    } else {
        // ====================================================================
        // Worker Process (Rank 1 ~ size-1): Receives subtasks, computes, 
        // and returns local results.
        // ====================================================================

        // -- Step 1: Receive the entire matrix B --
        std::vector<double> B(n * k); 
        MPI_Recv(B.data(), n * k, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // -- Step 2: Calculate the number of assigned rows and receive the local block of matrix A --
        int rows_per_worker = m / (size - 1);
        int start_row = (rank - 1) * rows_per_worker;
        int num_rows = (rank == size - 1) ? (m - start_row) : rows_per_worker;
        
        std::vector<double> local_A(num_rows * n);
        MPI_Recv(local_A.data(), num_rows * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // -- Step 3: Perform local matrix multiplication (local_A * B = local_C) --
        // Applying the i-k-j loop order to maximize spatial locality in memory
        std::vector<double> local_C(num_rows * k, 0.0);
        for(int i = 0; i < num_rows; ++i) {
            for(int p = 0; p < n; ++p) {
                double r = local_A[i * n + p];
                for(int j = 0; j < k; ++j) {
                    local_C[i * k + j] += r * B[p * k + j];
                }
            }
        }

        // -- Step 4: Send the computed local_C back to the Master --
        MPI_Send(local_C.data(), num_rows * k, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    // 3. Finalize MPI environment
    MPI_Finalize();
    return 0;
}
