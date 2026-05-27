# Lab 7: MPI并行应用（FFT）

## 目标
基于课程提供的 `fft_serial.cpp` 思路，使用 MPI 多进程实现并行 FFT，并验证可运行性与数值正确性。

## 实现说明
- 文件：`src/mpi_fft.cpp`
- 并行方式：按数据连续分块给各进程，每一层蝶形计算前使用 `MPI_Allgather` 同步全局向量，再由各进程更新自己的局部区间。
- 正确性验证：前向 FFT + 逆向 FFT 后，与原始输入比较均方根误差（RMS error）。

## 构建与运行
```bash
make
mpirun -np 4 ./bin/mpi_fft 4096 20
```

参数含义：
- 第1个参数 `N`：信号长度（必须是 2 的幂且能被进程数整除）
- 第2个参数 `nits`：性能测试迭代次数

## 备注
- 当前 macOS 环境无法原生安装 Valgrind/massif（该工具要求 Linux），因此 Lab7 第2部分内存采样建议在 Linux 环境执行。
