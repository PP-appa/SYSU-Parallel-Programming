# 并行程序设计与算法

> 中山大学（SYSU）并行程序设计与算法课程实验代码与报告合集

## 目录

- [项目概述](#项目概述)
- [实验列表](#实验列表)
- [依赖与构建](#依赖与构建)
- [lab6 快速示例](#lab6-快速示例)
- [贡献与许可](#贡献与许可)

## 项目概述

本仓库包含课程实验（lab0..lab6）的代码、可执行文件和实验报告（`report.md`）。每个实验目录通常包含：

- 源码（`src/`）
- 可执行文件（`bin/`，若已构建）
- 实验报告（`report.md`）
- 可选的测试脚本（`run_experiments.py`）

## 实验列表

- `lab0`：基础与缓存优化（矩阵乘法变换、循环展开、MKL 对比）
- `lab1`：MPI（点对点通信示例）
- `lab2`：MPI（集合通信示例）
- `lab3`：Pthreads（共享内存并行示例）
- `lab4`：Pthreads 加热板练习
- `lab5`：自定义 `parallel_for` 与 GEMM 对比
- `lab6`：Heated Plate（基于 Pthreads 的实现）

有关每个实验的详细说明、参数与结果，请查看对应目录下的 `report.md`。

## 依赖与构建

推荐在 Linux / WSL 环境下使用。常用依赖：

- 编译器：GCC (`g++`)
- 线程库：Pthreads
- 分布式（可选）：OpenMPI（用于 lab1、lab2）

常见编译选项示例：

```bash
# 常见示例（在对应 lab 子目录下执行）
g++ -std=c++11 -O2 -pthread -o bin/your_program src/your_program.cpp
```

## lab6 快速示例

以 `lab6` 为例，复现实验的基本步骤：

```bash
cd lab6
make all          # 构建可执行文件（需要 Makefile）
make run          # 使用 Makefile 中默认的运行配置
# 或手动运行，参数：<num_threads> <epsilon>
./bin/heated_plate_pthread 4 0.001
```

`run_experiments.py`（若存在）可以在不同线程数间自动化运行并保存结果（例如 1/2/4/8）。

