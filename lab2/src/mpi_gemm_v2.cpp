/**
 * @file mpi_gemm_v2.cpp
 * @brief MPI Matrix Multiplication using Collective Communication (Lab 2)
 * @details Replaces P2P communication (Send/Recv) with Broadcast/Scatter/Gather
 * and applies i-p-j loop ordering for cache locality optimization.
 */

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cstddef> // For offsetof

// 1. Structure definition for broadcasting multiple configuration variables at once
struct MatrixConfig {
    int m;
    int n;
    int k;
};

int main(int argc, char** argv) {
    // Initialize MPI environment
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MatrixConfig config = {128, 128, 128};
    if (rank == 0 && argc >= 4) {
        config.m = std::atoi(argv[1]);
        config.n = std::atoi(argv[2]);
        config.k = std::atoi(argv[3]);
    }

    // 2. Create custom MPI datatype for MatrixConfig struct
    MPI_Datatype mpi_config_type;
    int blocklengths[3] = {1, 1, 1}; // Number of items per block
    MPI_Aint displacements[3];       // Memory offsets for each struct member
    displacements[0] = offsetof(MatrixConfig, m);
    displacements[1] = offsetof(MatrixConfig, n);
    displacements[2] = offsetof(MatrixConfig, k);

    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT}; // MPI types for struct members

    // Register the custom datatype containing 3 integers
    MPI_Type_create_struct(3, blocklengths, displacements, types, &mpi_config_type);
    MPI_Type_commit(&mpi_config_type); // Commit the type before use

    // Broadcast the configuration struct from rank 0 to all processes
    MPI_Bcast(&config, 1, mpi_config_type, 0, MPI_COMM_WORLD);

    // Free the custom datatype after use
    MPI_Type_free(&mpi_config_type);

    // 3. Compute the number of rows assigned to each process (assuming divisibility for simplicity)
    int m = config.m, n = config.n, k = config.k;
    int local_m = m / size; 

    // 4. Allocate memory buffers for all processes
    std::vector<double> B(n * k);                  // Entire matrix B (needed by all)
    std::vector<double> local_A(local_m * n);      // Local row block of matrix A
    std::vector<double> local_C(local_m * k, 0.0); // Local computed row block of matrix C

    std::vector<double> full_A; 
    std::vector<double> full_C; 
    
    // Master process allocates space for the complete matrices A and C
    if (rank == 0) {
        full_A.resize(m * n);
        full_C.resize(m * k);
        for(int i = 0; i < m * n; ++i) full_A[i] = (double)rand() / RAND_MAX;
        for(int i = 0; i < n * k; ++i) B[i] = (double)rand() / RAND_MAX;
    }

    // 5. Data Distribution using MPI Collective Communication
    
    // Step 5.1: Broadcast the entire matrix B from rank 0 to all other processes
    MPI_Bcast(B.data(), n * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Step 5.2: Scatter the complete matrix full_A into smaller local_A blocks array for each process
    MPI_Scatter(full_A.data(), local_m * n, MPI_DOUBLE, 
                local_A.data(), local_m * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Synchronize all processes to start accurate timing
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // 6. Perform local matrix multiplication calculation (local_C = local_A * B)
    // Applying i-p-j loop ordering to maximize spatial cache locality (Row-major layout)
    for (int i = 0; i < local_m; ++i) {
        for (int p = 0; p < n; ++p) {
            for (int j = 0; j < k; ++j) {                
                local_C[i * k + j] += local_A[i * n + p] * B[p * k + j];
            }
        }
    }

    // 7. Results Collection using MPI Collective Communication
    // Gather all local_C results and append them seamlessly into the master's full_C
    MPI_Gather(local_C.data(), local_m * k, MPI_DOUBLE, 
               full_C.data(), local_m * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
               
    // Synchronize all processes before recording end time
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        std::cout << "m=" << m << ", n=" << n << ", k=" << k 
                  << " | Processes=" << size 
                  << " | Time=" << end_time - start_time << " seconds" << std::endl;
    }

    // Terminate MPI execution environment
    MPI_Finalize();
    return 0;
}
