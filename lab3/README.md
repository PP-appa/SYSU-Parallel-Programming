# Lab 3 - 基于 Pthreads 的并行矩阵乘法和数组求和

此目录包含并行程序设计与算法课程 Lab 3 的实现代码。该实验主要使用 POSIX 线程（Pthreads）标准在多核架构上实现并行执行。

## 结构

- **`src/pthread_gemm.cpp`**: 采用块划分执行的多线程矩阵乘法。
- **`src/pthread_sum.cpp`**: 并行数组求和，通过线程本地寄存器进行优化以避免伪共享（False Sharing）。
- **`src/pthread_sum_falsesharing.cpp`**: 用于教学目的的实现代码，故意导致缓存行失效（伪共享），以将其性能损失作为基准进行测试。
- **`report.md`**: 实验观察报告，记录了执行时间、加速比以及深度的架构分析。

## 编译

需要一个支持 `-lpthread` 的基本 C++ 环境（例如 Linux/WSL 上的 GCC）。从 `lab3` 的根目录运行以下命令：

```bash
mkdir -p bin

# 1. 编译并行矩阵乘法
g++ -O3 src/pthread_gemm.cpp -o bin/pthread_gemm -lpthread

# 2. 编译并行数组求和
g++ -O3 src/pthread_sum.cpp -o bin/pthread_sum -lpthread

# 3. 编译伪共享基准测试 
# (注意: 使用 -O0 以明确防止 GCC 优化掉伪共享行为)
g++ -O0 src/pthread_sum_falsesharing.cpp -o bin/pthread_sum_falsesharing_O0 -lpthread
g++ -O0 src/pthread_sum.cpp -o bin/pthread_sum_O0 -lpthread
```

## 运行示例

矩阵乘法需要四个参数：`<m> <n> <k> <num_threads>`。
```bash
./bin/pthread_gemm 2048 2048 2048 16
```

数组求和需要两个参数：`<array_size_n> <num_threads>`。
```bash
./bin/pthread_sum 128000000 16
./bin/pthread_sum_O0 128000000 16
./bin/pthread_sum_falsesharing_O0 128000000 16
```

你也可以找到自动化 Shell 脚本（如 `run_experiments.sh`）以顺序测试多个数据向量。
