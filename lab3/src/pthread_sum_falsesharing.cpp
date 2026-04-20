/**
 * @file pthread_sum_falsesharing.cpp
 * @brief Performance demonstration of False Sharing in Pthreads
 * 
 * This program serves as a comparative benchmark to parallel array summation.
 * It purposefully implements a naive shared memory writing approach 
 * where multiple CPU cores constantly invalidate each other's Cache line.
 * This hardware phenomenon is referred to as "False Sharing".
 */

#include <iostream>
#include <vector>
#include <pthread.h>
#include <chrono>
#include <random>

using namespace std;

/**
 * @struct SumData
 * @brief Arguments passed to the naive summation thread.
 */
struct SumData {
    int thread_id;
    int start_idx;
    int end_idx;
    long long* local_sum; 
    const int* A;
};

/**
 * @brief Naive worker function triggering False Sharing.
 * 
 * Frequently accesses and updates `*(data->local_sum)` inside the loop.
 * As neighboring threads are writing to adjacent indices of `local_sums` 
 * (which likely reside on the same 64-byte Cache line), intensive hardware
 * cache invalidation occurs, severely stalling the overall execution time.
 * 
 * @param arg Pointer to SumData
 * @return void* 
 */
void* sum_worker_false_sharing(void* arg) {
    SumData* data = (SumData*)arg;
    *(data->local_sum) = 0;
    
    for (int i = data->start_idx; i < data->end_idx; ++i) {
        // [ANTI-PATTERN] Directly accumulating data directly over a shared memory pointer.
        // This effectively writes main memory / L1 Cache constantly.
        *(data->local_sum) += data->A[i]; 
    }
    
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
    for (long long i = 0; i < n; ++i) A[i] = 1;

    auto start_time = chrono::high_resolution_clock::now();

    pthread_t* threads = new pthread_t[num_threads];
    SumData* thread_data = new SumData[num_threads];
    long long* local_sums = new long long[num_threads](); 

    long long chunk_size = n / num_threads;
    for (int i = 0; i < num_threads; ++i) {
        thread_data[i].thread_id = i;
        thread_data[i].A = A;
        thread_data[i].local_sum = &local_sums[i];
        
        thread_data[i].start_idx = i * chunk_size;
        thread_data[i].end_idx = (i == num_threads - 1) ? n : (i + 1) * chunk_size;

        pthread_create(&threads[i], NULL, sum_worker_false_sharing, (void*)&thread_data[i]);
    }

    long long total_sum = 0;
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
        total_sum += local_sums[i];
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    cout << "False Sharing - n: " << n << ", threads: " << num_threads 
         << ", sum: " << total_sum 
         << ", time: " << elapsed.count() << " s\n";

    delete[] A; delete[] threads; delete[] thread_data; delete[] local_sums;
    return 0;
}
