# Lab 4 - 基于 Pthreads 的条件变量机制与蒙特卡洛算法

此目录包含并行程序设计与算法课程 Lab 4 的实现代码。该实验侧重于使用 POSIX Threads (Pthreads) 标准中的条件变量解决任务依赖同步，并体现对于完全纯数据并行的扩展性。

## 结构

- **`src/quadratic.cpp`**: 并行一元二次方程求解。通过建立三个计算线程，利用 `pthread_cond_t` 让 $x_1, x_2$ 的求解过程等待判别式 $\Delta$ 的前置完成信号。
- **`src/monte_carlo.cpp`**: 并行蒙特卡洛法求 $\pi$ 值。利用多个线程并行进行概率投点并利用互斥锁 `pthread_mutex_t` 对最终统计数 $m$ 进行归约聚合。
- **`run_experiments.py`**: 用于自动化执行蒙特卡洛方法并发性能测试。脚本会自动分配不同的线程数量跑分，并生成数据对比情况表格。
- **`report.md`**: 学号对应的实验观察报告，记录了运行逻辑、执行时间、加速比以及并行的深入理论分析。

## 编译

需要一个支持 `-lpthread` 的基本 C++ 环境。从 `lab4` 的根目录运行以下命令进行编译操作：

```bash
mkdir -p bin

# 1. 编译一元二次方程求解
g++ -O3 -pthread src/quadratic.cpp -o bin/quadratic

# 2. 编译蒙特卡洛方法求解
g++ -O3 -pthread src/monte_carlo.cpp -o bin/monte_carlo
```

## 运行示例

### 1. 一元二次方求解
```bash
./bin/quadratic
# 会提示你输入参数: a, b, c，或者是直接采用默认参数进行一次性结算
```

### 2. 蒙特卡洛求近似 π
该程序除了可以读标准输入外，已被扩充为通过 CLI 接收参数（方便测速脚本）。参数结构：`<n_points> <num_threads>`。
```bash
# 例子：利用 4 个线程投 65536 个点
./bin/monte_carlo 65536 4
```

## 自动化测试
直接运行 Python 脚本，评估从 1 至 16 线程在1亿运算量级下的实际加速比情况：
```bash
python3 run_experiments.py
```
