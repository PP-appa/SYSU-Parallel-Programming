import subprocess
import re
import os

def run_cmd(cmd, cwd):
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    match = re.search(r"Time consumed t: ([\d\.]+) s", result.stdout)
    if match:
        return float(match.group(1))
    return None

def main():
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    # sizes = [512, 1024, 2048]
    sizes = [2048]
    threads_list = [1, 2, 4, 8]
    schedules = [0, 1, 2] # 0: default, 1: static, 2: dynamic
    schedule_names = ["Default", "Static(1)", "Dynamic(1)"]
    
    print("=== Task 1 & 2: OpenMP GEMM ===")
    print("| Size | Threads | Schedule | Time (s) |")
    print("|------|---------|----------|----------|")
    
    for size in sizes:
        for t in threads_list:
            for s in schedules:
                if (size == 2048 and t == 1): continue # 刚刚跑过了
                cmd = ["./bin/omp_gemm", str(size), str(t), str(s)]
                time_t = run_cmd(cmd, cwd)
                print(f"| {size} | {t} | {schedule_names[s]} | {time_t:.4f} |")
                
    print("\n=== Task 3: Custom Pthreads parallel_for GEMM ===")
    print("| Size | Threads | Time (s) |")
    print("|------|---------|----------|")
    sizes = [512, 1024, 2048]
    for size in sizes:
        for t in threads_list:
            cmd = ["./bin/custom_gemm", str(size), str(t)]
            time_t = run_cmd(cmd, cwd)
            print(f"| {size} | {t} | {time_t:.4f} |")

if __name__ == "__main__":
    main()
