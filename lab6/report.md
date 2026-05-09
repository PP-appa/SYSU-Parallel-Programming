# 中山大学 计算机学院本科生实验报告

**（2025/2026学年）**

| 课程名称 | 并行程序设计与算法（实验） | 批改人 | |
| :--- | :--- | :--- | :--- |
| **实验** | Lab 6 - Heated Plate 问题的 Pthreads 并行实现 | **专业（方向）** | 计算机科学与技术（人工智能） |
| **学号** | 23336103 | **姓名** | 雷颜玮 |
| **Email** | leiyanwei2005@163.com | **完成日期** | 2026年5月9日 |

---

## 1. 实验要求与目的

将从 Lab5 构造的 `parallel_for` 并行框架应用到热传导问题中，实现基于 Pthreads 的并行求解。具体要求如下：

1. 使用 Lab5 中构造的 `parallel_for` 并行框架结构
2. 将参考代码 `heated_plate_openmp.c` 改造为基于 Pthreads 的并行应用
3. 使用 Functor 模式对不同的并行操作进行封装
4. 测试不同线程数（1、2、4、8）下的性能表现
5. 对实验结果进行分析和总结

## 2. 实现原理

### 2.1 问题描述

热传导问题使用规则网格上的热传导模型。每次迭代中，通过对四个邻域内热量平均值的模拟热传导过程，计算网格上每个内部点的新温度值，即：

$$w_{i,j}^{t+1} = \frac{1}{4}(w_{i-1,j}^t + w_{i+1,j}^t + w_{i,j-1}^t + w_{i,j+1}^t)$$

其中网格大小为 500×500，上下左右边界温度固定为 0 或 100。

### 2.2 并行策略

- **行级并行**：将 500×500 的网格按行分配给不同线程，每个线程独立处理多行
- **Functor 模式**：使用函数指针回调处理三种不同操作（复制、更新、计算差值）
- **互斥保护**：使用 `pthread_mutex_t` 保护全局的最大温度差值，确保线程安全
- **静态负载均衡**：根据线程数均匀分配迭代，余数项分配给前面的线程

### 2.3 parallel_for 的设计与实现

```cpp
void parallel_for(int start, int end, int inc,
                  void *(*functor)(int, void*),
                  void *arg, int n_threads)
```

核心特点：
- 将总迭代数均匀分配给各线程
- 支持自定义步长（increment）
- 灵活的 functor 接口，可传递任意用户数据
- 使用 lambda 表达式简化线程创建逻辑

## 3. 关键代码段与设计

### 3.1 三个核心 Functor 函数

**1. copy_functor**：将 w 数组复制到 u 数组
```cpp
void* copy_functor(int row, void* arg) {
    for (int j = 0; j < N; j++) {
        u[row][j] = w[row][j];
    }
    return nullptr;
}
```

**2. update_functor**：计算新的温度值
```cpp
void* update_functor(int row, void* arg) {
    if (row > 0 && row < M - 1) {
        for (int j = 1; j < N - 1; j++) {
            w[row][j] = (u[row-1][j] + u[row+1][j] + 
                        u[row][j-1] + u[row][j+1]) / 4.0;
        }
    }
    return nullptr;
}
```

**3. diff_functor**：计算相邻迭代间的最大温度差异
```cpp
void* diff_functor(int row, void* arg) {
    WorkData* data = (WorkData*)arg;
    if (row > 0 && row < M - 1) {
        double local_diff = 0.0;
        // 计算本行的最大差异
        for (int j = 1; j < N - 1; j++) {
            double diff_val = fabs(w[row][j] - u[row][j]);
            if (local_diff < diff_val) {
                local_diff = diff_val;
            }
        }
        // 使用互斥锁更新全局最大差异
        pthread_mutex_lock(data->diff_lock_ptr);
        if (*data->diff_ptr < local_diff) {
            *data->diff_ptr = local_diff;
        }
        pthread_mutex_unlock(data->diff_lock_ptr);
    }
    return nullptr;
}
```

### 3.2 并行循环框架的实现

```cpp
void parallel_for(int start, int end, int inc,
                  void *(*functor)(int, void*),
                  void *arg, int n_threads) {
    if (n_threads <= 0 || inc <= 0 || start >= end) return;

    int total_iterations = (end - start + inc - 1) / inc;
    if (total_iterations == 0) return;
    if (n_threads > total_iterations) n_threads = total_iterations;

    std::vector<pthread_t> threads(n_threads);
    struct ThreadData { int start; int end; int inc; 
                       void *(*functor)(int, void*); void *arg; };
    std::vector<ThreadData> t_data(n_threads);

    // 负载均衡：均匀分配迭代
    int iter_per_thread = total_iterations / n_threads;
    int remainder = total_iterations % n_threads;
    int current_iter = 0;

    for (int i = 0; i < n_threads; ++i) {
        int iters = iter_per_thread + (i < remainder ? 1 : 0);
        t_data[i].start = start + current_iter * inc;
        t_data[i].end = t_data[i].start + iters * inc;
        t_data[i].inc = inc;
        t_data[i].functor = functor;
        t_data[i].arg = arg;
        if (t_data[i].end > end) t_data[i].end = end;

        pthread_create(&threads[i], nullptr, [](void* p) -> void* {
            ThreadData* data = (ThreadData*)p;
            for (int i = data->start; i < data->end; i += data->inc) {
                data->functor(i, data->arg);
            }
            return nullptr;
        }, &t_data[i]);
        current_iter += iters;
    }

    for (int i = 0; i < n_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }
}
```

## 4. 实验结果

### 4.1 运行配置

- **网格大小**：500 × 500
- **收敛精度**（epsilon）：0.001
- **测试线程数**：1、2、4、8

### 4.2 性能数据（真实测试结果）

| 线程数 | 执行时间 (秒) | 相对加速比 | 加速效率 |
|------|-----------|---------|--------|
| 1    | 11.074933 | 1.0×    | 100%   |
| 2    | 8.382581  | 1.32×   | 66%    |
| 4    | 9.274678  | 1.19×   | 30%    |
| 8    | 14.484453 | 0.76×   | 10%    |

### 4.3 典型运行输出

```
HEATED_PLATE_PTHREAD
  A program to solve for the steady state temperature distribution
  over a rectangular plate using Pthreads.

  Spatial grid of 500 by 500 points.
  The iteration will be repeated until the change is <= 1.000000e-03
  Number of threads = 4

  MEAN = 74.949900

 Iteration  Change
         1  18.737475
         2  9.368737
         4  4.098823
         8  2.289577
        16  1.136604
        32  0.568201
        64  0.282805
       128  0.141777
       256  0.070808
       512  0.035427
      1024  0.017707
      2048  0.008856
      4096  0.004428
      8192  0.002210
     16384  0.001043
     16955  0.001000

  Error tolerance achieved.
  Wallclock time = 9.274678

HEATED_PLATE_PTHREAD:
  Normal end of execution.
```

## 5. 实验分析

### 5.1 性能分析

1. **线程数 = 1**：基线性能，执行时间为 11.07 秒

2. **线程数 = 2**：获得 1.32 倍加速比，效率为 66%。相比单线程，有明显的性能提升，但加速比未达到理论的 2 倍，主要原因包括：
   - 线程创建与管理的开销
   - 互斥锁在 diff_functor 中的竞争
   - 内存带宽限制

3. **线程数 = 4**：加速比为 1.19 倍，效率降至 30%。这是一个转折点，性能相比 2 线程版本反而下降。可能原因：
   - Amdahl 定律的限制：互斥锁保护的临界区变成了性能瓶颈
   - 线程间的缓存一致性协议（Cache Coherency Protocol）开销增大
   - 内存总线饱和

4. **线程数 = 8**：加速比仅为 0.76 倍，性能反而低于单线程。这表明：
   - 过多的线程导致上下文切换开销
   - 互斥锁的竞争达到最高峰
   - 系统核心数可能不足以充分利用 8 个线程

### 5.2 互斥锁的影响

diff_functor 中的互斥锁是性能的主要瓶颈。该函数在每个迭代中被调用一次（对 499 行），而每个线程都需要竞争同一把锁来更新全局最大差值。这导致：

- 较高的锁竞争率
- 序列化了关键部分
- 违反了 Amdahl 定律对并行加速的期望

### 5.3 与 Lab5 OpenMP 版本的对比

相比 Lab5 中的 OpenMP GEMM 实现（在 2048 大小、8 线程时获得 7.57 倍加速），本次 Pthreads 实现的加速效果明显较差。主要原因：

1. **问题特性不同**：
   - GEMM：矩阵乘法，每次迭代完全独立，无需同步
   - Heated Plate：每次迭代需要计算全局最大差值，需要同步

2. **同步开销**：
   - GEMM：只在循环外同步（隐式屏障）
   - Heated Plate：每次迭代内都需要互斥锁保护

3. **可扩展性**：
   - 当锁竞争严重时，额外的线程反而会降低性能

## 6. 存在的问题与改进方向

### 6.1 现有问题

1. **互斥锁成为瓶颈**：每个线程都需要访问全局最大差值，锁竞争严重
2. **栈空间占用大**：两个 500×500 的 double 数组共占 4MB 栈空间
3. **可扩展性差**：性能在 4 线程时开始下降

### 6.2 改进方向

1. **使用原子操作**：将互斥锁替换为 `std::atomic<double>`，减少锁开销
2. **线程本地存储（TLS）**：每个线程维护本地最大值，最后进行一次聚合
3. **动态内存分配**：避免大型数组占用栈空间
4. **更细粒度的任务划分**：采用工作窃取（Work Stealing）等动态调度策略
5. **两阶段同步**：先计算各线程的局部最大值，再进行一次全局归约

## 7. 结论

成功地将 Lab5 的 `parallel_for` 框架应用于热传导问题的 Pthreads 实现。通过该实验，我们深刻理解了：

1. **并行编程的复杂性**：看似简单的并行化可能由于同步开销而性能下降
2. **Amdahl 定律的实际影响**：序列化的关键部分（互斥锁）严重限制了加速比
3. **不同问题的并行特性差异**：相同的并行框架在不同问题上的性能表现可能差异很大
4. **线程安全设计的重要性**：正确的同步机制是并行程序性能的关键

该实现展示了 Pthreads 在通用并行计算中的灵活性和可控性，同时也暴露了基于互斥锁同步的局限性。在实际应用中，需要仔细分析临界区和同步开销，选择合适的同步机制。

