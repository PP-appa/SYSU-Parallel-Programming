/**
 * @file pthread_gemm.cpp
 * @brief Parallel Matrix Multiplication using Pthreads
 * 
 * This program implements a multi-threaded matrix multiplication (C = A * B)
 * using Block Data Distribution. Matrix rows are evenly distributed among threads 
 * to ensure load balancing and avoid write conflicts.
 */

#include <iostream>
#include <vector>
#include <pthread.h>
#include <chrono>
#include <random>

using namespace std;

/**
 * @struct ThreadData
 * @brief Structure to pass multidimensional parameters to Pthread workers.
 */
struct ThreadData {
    int thread_id;
    int start_row;
    int end_row;
    int m, n, k;
    double *A, *B, *C;
};

/**
 * @brief Worker function for parallel matrix multiplication.
 * 
 * Computes a distinct contiguous subset of rows for the resultant matrix C.
 * 
 * @param arg Pointer to ThreadData structure.
 * @return void* 
 */
void* gemm_worker(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    
    for (int i = data->start_row; i < data->end_row; ++i) {
        for (int k = 0; k < data->k; ++k) {
            double temp = data->A[i * data->k + k]; 
            for (int j = 0; j < data->n; ++j) {
                data->C[i * data->n + j] += temp * data->B[k * data->n + j];
            }
        }
    }
    
    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    if (argc != 5) {
        cout << "Usage: " << argv[0] << " <m> <n> <k> <num_threads>\n";
        return 1;
    }

    int m = atoi(argv[1]), n = atoi(argv[2]), k = atoi(argv[3]), num_threads = atoi(argv[4]);

    double* A = new double[m * n];
    double* B = new double[n * k];
    double* C = new double[m * k](); 

    // Initialize matrices with dummy values for benchmark purposes
    for (int i = 0; i < m * n; ++i) A[i] = 1.0;
    for (int i = 0; i < n * k; ++i) B[i] = 1.0;

    auto start_time = chrono::high_resolution_clock::now();

    pthread_t* threads = new pthread_t[num_threads];
    ThreadData* thread_data = new ThreadData[num_threads];

    // Thread creation and job scheduling (Block block partition)
    int chunk_size = m / num_threads;
    for (int i = 0; i < num_threads; ++i) {
        thread_data[i].thread_id = i;
        thread_data[i].m = m; thread_data[i].n = n; thread_data[i].k = k;
        thread_data[i].A = A; thread_data[i].B = B; thread_data[i].C = C;
        
        thread_data[i].start_row = i * chunk_size;
        thread_data[i].end_row = (i == num_threads - 1) ? m : (i + 1) * chunk_size;

        pthread_create(&threads[i], NULL, gemm_worker, (void*)&thread_data[i]);
    }

    // Thread synchronization
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    cout << "m: " << m << ", n: " << n << ", k: " << k << ", threads: " << num_threads 
         << ", time: " << elapsed.count() << " s\n";

    delete[] A; delete[] B; delete[] C;
    delete[] threads; delete[] thread_data;

    return 0;
}
