import subprocess
import re

processes = [1, 2, 4, 8, 16]
sizes = [128, 256, 512, 1024, 2048]
results = {p: {s: None for s in sizes} for p in processes}
executable = "./bin/mpi_gemm_v2"

print("开始自动运行实验，这可能需要几分钟时间...")

for p in processes:
    for s in sizes:
        cmd = ["mpirun", "--oversubscribe", "-np", str(p), executable, str(s), str(s), str(s)]
        print(f"正在运行: {' '.join(cmd)}")
        try:
            output = subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.STDOUT)
            match = re.search(r"Time=([\d\.]+)", output)
            if match:
                results[p][s] = float(match.group(1))
            else:
                results[p][s] = "Error"
        except Exception:
            results[p][s] = "Failed"

print("\n--- 测试完成，实验表格数据如下 ---\n")
header = "| 进程数 \\\\ 矩阵规模 | " + " | ".join([str(s) for s in sizes]) + " |"
separator = "|---" * (len(sizes) + 1) + "|"
print(header)
print(separator)

for p in processes:
    row = f"| **{p}** | "
    row += " | ".join([(f"{results[p][s]:.6f}" if isinstance(results[p][s], float) else str(results[p][s])) for s in sizes]) + " |"
    print(row)
