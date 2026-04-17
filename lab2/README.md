# Lab 2: 改进的 MPI 并行矩阵乘法 (MPI-v2)

## 实验目的
改进上次实验中点对点通信的 MPI 矩阵乘法，改用 **MPI 集合通信**（Collective Communication）来实现进程间的数据交互，并应用**软件缓存（Cache）局部性优化**以提升计算性能。

## 核心特性
1. **聚合通信变量**：使用 `MPI_Type_create_struct` 将矩阵维度 `m, n, k` 聚合成一个结构体，并使用 `MPI_Bcast` 进行一次性广播，替代了多次发送或强行拼凑数组的做法。
2. **集合通信分发数据**：舍弃繁琐的 `for` 循环与 `MPI_Send` / `MPI_Recv`，改用高效的 `MPI_Bcast` 广播只读矩阵 B，并用 `MPI_Scatter` 将矩阵 A 切块平分。
3. **缓存局部性优化**：在局部的核心矩阵乘法计算中，将常规的 `i-j-p` 三层循环调整为了 `i-p-j` 循环，使得内层计算严格沿着内存按行递增，大幅降低 Cache Miss。
4. **集合通信收集结果**：使用 `MPI_Gather` 高效且优雅地将各个进程的计算块合并回主进程的矩阵 C 中。

## 编译与运行

请确认当前处于 `lab2` 目录下，并且已有 `bin/` 文件夹。

### 1. 手动编译
```bash
mpicxx src/mpi_gemm_v2.cpp -o bin/mpi_gemm_v2 -O3
```

### 2. 自动化脚本运行与测试
使用 Python 脚本全自动运行 `[128, 256, 512, 1024, 2048]` 规模与 `[1, 2, 4, 8, 16]` 进程数的组合，并自动输出测试 Markdown 表格：
```bash
python3 run_experiments.py
```
*(注意：16 进程部分已在脚本中自动加入 `--oversubscribe` 以适应 CPU 核心数少于 16 时的限制)*
