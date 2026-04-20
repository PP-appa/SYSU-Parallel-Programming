/**
 * @file pthread_sum.cpp
 * @brief Parallel Array Summation using Pthreads
 * 
 * Demonstrates a parallel reduction algorithm using multi-threading.
 * To avoid the performance penalty of False Sharing caused by multiple 
 * threads concurrently writing to contiguous memory blocks, this implementation
 * securely accumulates data into a CPU-local register/variable before a single
 * memory write-back (Reduction phase).
 */

#include <iostream>
#include <vector>
#include <pthread.h>
#include <chrono>
#include <random>

using namespace std;

/**
 * @struct SumData
 * @brief Arguments passed to summation thread.
 */
struct SumData {
    int thread_id;
    int start_idx;
    int end_idx;
    long long* local_sum; // Pointer to thread's dedicated slot in the shared sums array
    const int* A;
};

/**
 * @brief Thread worker function for array summation.
 * 
 * Uses a local summation variable to prevent cache line invalidation
 * commonly known as False Sharing.
 * 
 * @param arg Pointer to SumData structure
 * @return void* 
 */
void* sum_worker(void* arg) {
    SumData* data = (SumData*)arg;
    
    long long sum = 0; // Use a local variable to accumulate (likely allocated in register)
    for (int i = data->start_idx; i < data->end_idx; ++i) {
        sum += data->A[i];
    }
    
    // Perform a single atomic-like write-back to the main memory once computations are done
    *(data->local_sum) = sum; 
    
    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <n> <num_threads>\n";
        return 1;
    }

    long long n = atoll(argv[1]);
    int num_threads = atoi(argv[2]);

    int* A = new int[n];

    // Initialize array with default values to verify results logic
    for (long long i = 0; i < n; ++i) A[i] = 1;

    auto start_time = chrono::high_resolution_clock::now();

    pthread_t* threads = new pthread_t[num_threads];
    SumData* thread_data = new SumData[num_threads];
    
    // Array to hold individual computation results of each thread (Reduction placeholder)
    long long* local_sums = new long long[num_threads](); 

    long long chunk_size = n / num_threads;
    for (int i = 0; i < num_threads; ++i) {
        thread_data[i].thread_id = i;
        thread_data[i].A = A;
        
        // Pass the designated pointer location avoiding data races
        thread_data[i].local_sum = &local_sums[i];
        
        thread_data[i].start_idx = i * chunk_size;
        thread_data[i].end_idx = (i == num_threads - 1) ? n : (i + 1) * chunk_size;

        pthread_create(&threads[i], NULL, sum_worker, (void*)&thread_data[i]);
    }

    // Global Reduction Phase
    long long total_sum = 0;
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
        total_sum += local_sums[i]; // Safely accumulate from individual slots
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    cout << "n: " << n << ", threads: " << num_threads 
         << ", sum: " << total_sum 
         << ", time: " << elapsed.count() << " s\n";

    delete[] A; delete[] threads; delete[] thread_data; delete[] local_sums;

    return 0;
}
