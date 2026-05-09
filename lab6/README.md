# Lab 6: Heated Plate Problem using Pthreads

## 目标

使用从 Lab5 构造的 `parallel_for` 并行结构，结合 Pthreads，将热传导问题从 OpenMP 实现改造为基于 Pthreads 的并行应用。

## 问题描述

热传导问题（Heated Plate）使用规则网格上的热传导模型。每次迭代中，通过对四个邻域内热量平均值的模拟热传导过程，即：

$$w_{i,j}^{t+1} = \frac{1}{4}(w_{i-1,j}^t + w_{i+1,j}^t + w_{i,j-1}^t + w_{i,j+1}^t)$$

其中网格大小为 500×500。

## 实现方法

### 核心机制

1. **parallel_for 函数**：
   - 基于 pthread 实现的并行循环结构
   - 将任务均匀分配给指定数量的线程
   - 支持自定义的 functor 回调函数

2. **三个主要操作**：
   - **copy_functor**：复制 w 数组到 u 数组
   - **update_functor**：计算新的温度值
   - **diff_functor**：计算相邻迭代间的最大差值（使用互斥锁保护）

3. **线程安全**：
   - 使用 `pthread_mutex_t` 保护全局的差值变量
   - 每个线程计算局部最大差值，然后通过互斥锁原子地更新全局最大差值

## 编译与运行

### 编译
```bash
make all
# 或直接编译
g++ -std=c++11 -pthread -O2 -o bin/heated_plate_pthread src/heated_plate_pthread.cpp
```

### 运行
```bash
# 使用 4 个线程（默认）
./bin/heated_plate_pthread 4 0.001

# 使用 1 个线程
./bin/heated_plate_pthread 1 0.001

# 使用 2 个线程
./bin/heated_plate_pthread 2 0.001

# 使用 8 个线程
./bin/heated_plate_pthread 8 0.001
```

### 使用 Makefile
```bash
make run              # 运行默认（4线程）
make run_1            # 运行 1 线程
make run_2            # 运行 2 线程
make run_4            # 运行 4 线程
make run_8            # 运行 8 线程
make test             # 运行所有版本测试
make clean            # 清理编译文件
```

## 参数说明

程序接受两个命令行参数：

1. **num_threads**：使用的线程数（默认 4）
2. **epsilon**：收敛精度，当相邻迭代的最大差值小于此值时停止迭代（默认 0.001）

## 输出说明

程序输出包括：
- 网格大小、精度要求和线程数配置
- 边界温度的平均值
- 每次迭代的次数和改变量
- 达到收敛时的总迭代次数
- 计算总耗时（秒）

## 性能特性

### parallel_for 设计
- 静态负载均衡：根据线程数均匀分配迭代
- 支持步长（increment）
- 灵活的 functor 接口，便于传递自定义操作

### 关键改进
- 使用 lambda 表达式简化线程创建
- 互斥锁保护临界区，确保线程安全
- 高效的负载分配策略

## 文件结构

```
lab6/
├── README.md                          # 项目说明
├── report.md                          # 实验报告
├── Makefile                           # 构建文件
├── src/
│   └── heated_plate_pthread.cpp      # 主程序
├── bin/
│   └── heated_plate_pthread          # 编译后的可执行文件
└── Heated-plate-参考资料/
    └── heated_plate_openmp.c         # 参考实现（OpenMP）
```

## 与原始 OpenMP 版本的对比

| 特性 | OpenMP | Pthreads |
|------|--------|----------|
| 并行化粒度 | 编译指令 | 显式函数调用 |
| 同步方式 | 隐式屏障 | pthread_join |
| 负载均衡 | 自动 | 手动分配 |
| 代码复杂度 | 较低 | 较高 |
| 灵活性 | 较低 | 较高 |

## 注意事项

- 程序使用了大量栈空间（500×500×8 字节×2 = 4MB），如遇栈溢出可考虑动态分配
- 互斥锁有一定的开销，在高频调用的情况下可考虑使用原子操作或线程本地存储优化
- 实际性能取决于系统的核心数和其他负载
