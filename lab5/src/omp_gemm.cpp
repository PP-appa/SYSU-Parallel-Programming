#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>
#include <string>

using namespace std;

void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main(int argc, char* argv[]) {
    int size = 512;
    int num_threads = 4;
    int schedule_type = 0; // 0: default, 1: static 1, 2: dynamic 1

    if (argc >= 2) size = atoi(argv[1]);
    if (argc >= 3) num_threads = atoi(argv[2]);
    if (argc >= 4) schedule_type = atoi(argv[3]);

    int M = size, N = size, K = size;
    float* A = new float[M * N];
    float* B = new float[N * K];
    float* C = new float[M * K];

    init_matrix(A, M, N);
    init_matrix(B, N, K);
    for (int i = 0; i < M * K; ++i) C[i] = 0.0f;

    omp_set_num_threads(num_threads);

    auto start_time = chrono::high_resolution_clock::now();

    if (schedule_type == 0) {
        // Default schedule
        #pragma omp parallel for
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                float a_ik = A[i * K + k];
                for (int j = 0; j < N; ++j) {
                    C[i * N + j] += a_ik * B[k * N + j];
                }
            }
        }
    } else if (schedule_type == 1) {
        // Static schedule
        #pragma omp parallel for schedule(static, 1)
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                float a_ik = A[i * K + k];
                for (int j = 0; j < N; ++j) {
                    C[i * N + j] += a_ik * B[k * N + j];
                }
            }
        }
    } else if (schedule_type == 2) {
        // Dynamic schedule
        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                float a_ik = A[i * K + k];
                for (int j = 0; j < N; ++j) {
                    C[i * N + j] += a_ik * B[k * N + j];
                }
            }
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    cout << "Time consumed t: " << elapsed.count() << " s" << endl;

    delete[] A; delete[] B; delete[] C;
    return 0;
}
