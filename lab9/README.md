# Lab 9: CUDA 矩阵转置

本目录按实验文档 `CUDA Hello World` 和 `CUDA矩阵转置` 搭好 CUDA 项目框架。当前本地环境没有 GPU，因此只做静态测试；请在服务器上使用 `nvcc` 构建并运行。

## 文件

- `src/cuda_hello.cu`：输入 `n m k`，启动 `n` 个线程块，每个线程块维度为 `m x k`，每个线程输出块编号、二维线程编号和 Hello World。
- `src/matrix_transpose.cu`：随机生成 `n x n` 单精度矩阵，支持 `naive` 全局内存转置和 `tiled` shared-memory 转置，输出时间、带宽估算和最大误差。
- `run_experiments.py`：批量扫描矩阵规模、线程块大小、kernel 类型，结果写入 `results/transpose_summary.csv`。
- `tests/test_lab9_static.py`：本地无 GPU 时使用的项目结构静态测试。
- `并行程序设计_23336103_雷颜玮.pdf`：最终提交报告，已根据 `report.md` 生成。

## 构建

```bash
cd lab9
make all
```

如果服务器上的 CUDA 编译器不叫 `nvcc`，可以指定：

```bash
make all NVCC=/path/to/nvcc
```

## CUDA Hello World

```bash
./bin/cuda_hello 2 4 4
```

参数含义：

- `n`：线程块数量，范围 `[1, 32]`
- `m`：线程块 x 方向线程数，范围 `[1, 32]`
- `k`：线程块 y 方向线程数，范围 `[1, 32]`

GPU 线程的输出顺序通常没有稳定规律。不同 block 和 warp 的调度、printf 缓冲刷新顺序都会影响最终显示顺序；主机端输出因为在 kernel 启动前执行，所以会先出现。

## 单次矩阵转置

```bash
./bin/matrix_transpose --n 1024 --kernel tiled --block-x 32 --block-y 8 --repeats 20
./bin/matrix_transpose --n 1024 --kernel naive --block-x 16 --block-y 16 --repeats 20
```

输出示例：

```text
matrix_size=1024 kernel=tiled block_x=32 block_y=8 repeats=20 time_ms=0.123456 bandwidth_gb_s=67.890000 max_abs_error=0.000000
```

其中 `bandwidth_gb_s` 按每次转置读写各一次矩阵估算，即 `2 * n * n * sizeof(float) / time`。

## 批量实验

```bash
python3 run_experiments.py
```

默认测试：

- 矩阵规模：`512, 1024, 2048`
- kernel：`naive, tiled`
- block：`16x16, 32x8, 32x16`
- 每组 kernel 重复：`20`

可自定义参数：

```bash
python3 run_experiments.py --sizes 512,768,1024,2048 --blocks 16x16 32x8 32x16 --kernels naive tiled --repeats 50
```

## 本地静态测试

```bash
make test
```

该测试不需要 GPU，只检查代码框架和实验脚本字段；真正的正确性和性能请以服务器上的 `make all`、单次运行和批量实验为准。

## 提交说明

仓库中的 `lab9` 目录已包含本实验需要的源码、实验结果和最终报告 PDF，可直接作为提交材料使用。
