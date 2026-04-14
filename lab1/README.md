# Lab 1: MPI 点对点通信与并行矩阵乘法 (MPI-v1)

## 实验简介
本实验基于 **Message Passing Interface (MPI)** 的基本点对点通信手段（即 `MPI_Send` 与 `MPI_Recv`），使用 C++ 从零实现了一个多进程、分布式的并行通用矩阵计算（GEMM）。

通过本实验：
1. 深入理解分布式内存编程中基础的数据交互与通讯原理。
2. 掌握经典的 **Master-Worker 调度模型**与矩阵行级切分方法。
3. 对真实进程数与不同矩阵规模下的加速比、临界通信开销等进行了性能探讨。
4. 吸取了 [Lab 0](../lab0) 的理论知识，在并行计算的核心逻辑中叠加了对底层 Cache Locality (`i-k-j` 循环) 优化。

## 编译与运行
依赖环境：`OpenMPI` 或 `MPICH` 工具链。

### 1. 编译
在项目的 `lab1` 目录下执行如下指令，将自动产生开启最高优化的可执行文件至 `bin/`。
```bash
mpicxx ./src/mpi_gemm.cpp -o ./bin/mpi_gemm -O3
```

### 2. 运行单次实验
通过 `mpirun` 指定所需的进程数并传入 $M, N, K$ 三个矩阵规模参数：
```bash
# 例如：使用 4 进程对规模为 1024x1024x1024 的矩阵进行并行乘法
mpirun -np 4 ./bin/mpi_gemm 1024 1024 1024
```
*(注：如果实验机硬件核心数有限且需要运行超过可用限制的进程，需在后面追加 `--oversubscribe` 参数。)*

### 3. 自动测试脚本
提供了一个一键化 Python 性能测试脚本 `run_experiments.py`。该脚本自动遍历从 $128 \sim 2048$ 的各种方阵维度以及 $1 \sim 16$ 种不同进程数量的交叉对比，最终输出一张支持 Markdown 语法的性能透视表格。
```bash
python3 run_experiments.py
```

## 实验内容导航
* `/src/mpi_gemm.cpp`: 实现核心逻辑的并行矩阵乘 C++ 源码。
* `run_experiments.py`: 批量测试脚本。
* `report.md`: 实验性能表单、分析数据，以及对于“超大型”和“超稀疏大型”阵列的算法设计与优化思考解答。
