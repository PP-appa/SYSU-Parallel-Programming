import subprocess
import re

processes = [1, 2, 4, 8, 16]
sizes = [128, 256, 512, 1024, 2048]
executable = "/home/yanweilei/parallel_projects/lab1/bin/mpi_gemm"

# 结果矩阵字典
results = {p: {s: 0.0 for s in sizes} for p in processes}

print("开始自动性能测试...")

for p in processes:
    for s in sizes:
        cmd = ["mpirun", "--oversubscribe", "-np", str(p), executable, str(s), str(s), str(s)]
        print(f"Running: {' '.join(cmd)}")
        try:
            # 运行命令截取输出
            run_result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = run_result.stdout
            
            # 正则提取时间数据
            match = re.search(r"Time:\s+([0-9\.]+)\s+seconds", output)
            if match:
                results[p][s] = float(match.group(1))
            else:
                print(f"Warning: 解析失败, 输出内容: {output}")
        except subprocess.CalledProcessError as e:
            print(f"Error: 运行失败 {e}")

print("\n测试完成！你可以直接把下面的表格复制到你的 report.md 中：\n")
print("| 进程数 | 128 | 256 | 512 | 1024 | 2048 |")
print("| :---: | :---: | :---: | :---: | :---: | :---: |")
for p in processes:
    row = f"| {p} |"
    for s in sizes:
        row += f" {results[p][s]:.6f} |"
    print(row)
