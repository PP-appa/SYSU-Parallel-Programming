/**
 * @file mpi_gemm.cpp
 * @brief 使用 MPI 点对点通信实现的并行通用矩阵乘法 (Lab 1)
 * @details 采用典型的 Master-Worker 模式。为了便于 MPI 通道连续传输，
 * 矩阵数据采用一维 std::vector 结构（Row-major 布局）进行模拟。
 */

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

int main(int argc, char** argv) {
    // 1. 初始化 MPI 环境
    MPI_Init(&argc, &argv);

    int rank, size;
    // 获取当前进程的 rank 编号
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    // 获取参与通信的总进程数
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    // 2. 解析矩阵规模参数 (默认 m=128, n=128, k=128)
    int m = 128, n = 128, k = 128;
    if (argc >= 4) {
        m = std::atoi(argv[1]);
        n = std::atoi(argv[2]);
        k = std::atoi(argv[3]);
    }

    if (rank == 0) {
        // ====================================================================
        // Master 进程 (Rank 0): 负责矩阵的初始化、任务的分发和结果的收集拼装
        // ====================================================================
        
        std::vector<double> A(m * n);
        std::vector<double> B(n * k);
        std::vector<double> C(m * k, 0.0);

        // 随机数初始化矩阵 A 和 B
        std::srand(static_cast<unsigned>(std::time(nullptr)));
        for (int i = 0; i < m * n; ++i) A[i] = (double)rand() / RAND_MAX;
        for (int i = 0; i < n * k; ++i) B[i] = (double)rand() / RAND_MAX;

        double start_time = MPI_Wtime();

        if (size > 1) {
            // -- 阶段 1: 广播全量矩阵 B 给所有 Worker --
            for (int dest = 1; dest < size; ++dest) {
                MPI_Send(B.data(), n * k, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            }

            // -- 阶段 2: 按行划分矩阵 A 并分发给所有 Worker --
            int rows_per_worker = m / (size - 1);
            for (int dest = 1; dest < size; ++dest) {
                int start_row = (dest - 1) * rows_per_worker;
                // 余数行全部交给最后一个 Worker 处理
                int num_rows = (dest == size - 1) ? (m - start_row) : rows_per_worker; 
                MPI_Send(A.data() + start_row * n, num_rows * n, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            }

            // -- 阶段 3: 接收 Worker 的局部计算结果并拼接至全局矩阵 C --
            for (int src = 1; src < size; ++src) {
                int start_row = (src - 1) * rows_per_worker;
                int num_rows = (src == size - 1) ? (m - start_row) : rows_per_worker;
                // 直接利用目标内存偏移量接收数据，避免额外拷贝
                MPI_Recv(C.data() + start_row * k, num_rows * k, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

        } else {
            // 当 size == 1 时，由 Master 亲自完成串行计算 (防除 0 和死锁)
            // 采用 i-k-j 循环顺序以优化 Cache 命中率
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
        // Worker 进程 (Rank 1 ~ size-1): 负责接收子任务进行计算并返回
        // ====================================================================

        // -- 步骤 1: 接收全图矩阵 B --
        std::vector<double> B(n * k); 
        MPI_Recv(B.data(), n * k, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // -- 步骤 2: 计算需分配给当前 Worker 的行数，并接收其专属的矩阵 A 子块 --
        int rows_per_worker = m / (size - 1);
        int start_row = (rank - 1) * rows_per_worker;
        int num_rows = (rank == size - 1) ? (m - start_row) : rows_per_worker;
        
        std::vector<double> local_A(num_rows * n);
        MPI_Recv(local_A.data(), num_rows * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // -- 步骤 3: 执行局部的矩阵乘法 local_A * B = local_C --
        // 使用 i-k-j 循环顺序深度挖掘数组读取时的内存空间局部性
        std::vector<double> local_C(num_rows * k, 0.0);
        for(int i = 0; i < num_rows; ++i) {
            for(int p = 0; p < n; ++p) {
                double r = local_A[i * n + p];
                for(int j = 0; j < k; ++j) {
                    local_C[i * k + j] += r * B[p * k + j];
                }
            }
        }

        // -- 步骤 4: 将算得的局部 local_C 返回送给 Master --
        MPI_Send(local_C.data(), num_rows * k, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    // 3. 释放 MPI 资源并正常退出
    MPI_Finalize();
    return 0;
}
