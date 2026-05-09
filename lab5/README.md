# Lab 5 - 基于 OpenMP 的并行矩阵乘法与自定义并行动态库

此目录包含并行程序设计与算法课程 Lab 5 的实现代码。主要展示如何应用 OpenMP 和 自定义 `parallel_for` (底层封装 POSIX Threads) 进行矩阵计算级别的加速扩展与任务调度优化。

## 结构

- **`src/omp_gemm.cpp`**: 任务一及二。使用 OpenMP 将传统串行矩阵乘法改造为并行化代码。支持按参数分别选择不同的切片调度模式（`default`, `schedule(static,1)`, `schedule(dynamic,1)`）。
- **`src/parallel_for.h`, `src/parallel_for.cpp`**: 任务三。依据 OpenMP 的思想重新基于原生 Pthreads 的多线程并行任务进行动态分配处理器的自造轮子 `libparallelfor.so` 动态库。
- **`src/custom_gemm.cpp`**: 任务三配套矩阵乘法。对以上编写封装完毕的自定义库进行引入使用。
- **`lib/`**: 存放编译出的自定义动态链接库。

## 编译运行
所需环境：含 `-fopenmp` 和 `-lpthread` 的 Linux C++ 基础环境。
在 `lab5` 目录下可以直接使用以下命令编译出二进制及链接库：
```bash
mkdir -p bin lib

# 编译 OpenMP 程序
g++ -O3 -fopenmp src/omp_gemm.cpp -o bin/omp_gemm

# 编译动态链接库 (Dynamic lib) 与配套的自定义函数程序
g++ -fPIC -shared -O3 -pthread src/parallel_for.cpp -o lib/libparallelfor.so
g++ -O3 src/custom_gemm.cpp -Llib -lparallelfor -Wl,-rpath,./lib -o bin/custom_gemm
```

## 测试脚本
为了得出报告中规模矩阵下的横向对比跑分，提供了现成的测试脚本测定吞吐量：
```bash
python3 run_experiments.py
```
