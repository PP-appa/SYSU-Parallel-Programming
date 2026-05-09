#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "parallel_for.h"

using namespace std;

// 用于传递给 functor 的参数块
struct functor_args {
    float *A;
    float *B;
    float *C;
    int M;
    int N;
    int K;
};

// 每次循环所执行的内容 (外层循环的 body)
// 传入的 idx 对应外层循环的行号 i
void* gemm_functor(int idx, void* args) {
    functor_args* data = (functor_args*)args;
    int i = idx;
    int K = data->K;
    int N = data->N;
    
    for (int k = 0; k < K; ++k) {
        float a_ik = data->A[i * K + k];
        for (int j = 0; j < N; ++j) {
            data->C[i * N + j] += a_ik * data->B[k * N + j];
        }
    }
    return nullptr;
}

void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main(int argc, char* argv[]) {
    int size = 512;
    int num_threads = 4;

    if (argc >= 2) size = atoi(argv[1]);
    if (argc >= 3) num_threads = atoi(argv[2]);

    int M = size, N = size, K = size;
    float* A = new float[M * N];
    float* B = new float[N * K];
    float* C = new float[M * K];

    init_matrix(A, M, N);
    init_matrix(B, N, K);
    for (int i = 0; i < M * K; ++i) C[i] = 0.0f;

    functor_args args = {A, B, C, M, N, K};

    auto start_time = chrono::high_resolution_clock::now();

    // 调用我们在动态链接库里手写的 parallel_for
    parallel_for(0, M, 1, gemm_functor, (void*)&args, num_threads);

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    cout << "Time consumed t: " << elapsed.count() << " s" << endl;

    delete[] A; delete[] B; delete[] C;
    return 0;
}
