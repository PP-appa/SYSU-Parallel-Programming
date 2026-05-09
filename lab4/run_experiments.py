import subprocess
import re
import os

def run_monte_carlo(n, num_threads):
    cmd = [f"./bin/monte_carlo", str(n), str(num_threads)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    output = result.stdout
    match = re.search(r"Time consumed t: ([\d\.]+) s", output)
    if match:
        return float(match.group(1))
    return None

def main():
    threads_list = [1, 2, 4, 8, 16]
    n = 100000000  # 10^8 points
    
    print(f"Running Monte Carlo with N = {n}...")
    times = []
    
    for t in threads_list:
        time_t = run_monte_carlo(n, t)
        times.append(time_t)
        
    speedups = [times[0] / t if t else 0 for t in times]
    efficiency = [speedups[i] / threads_list[i] for i in range(len(threads_list))]
    
    print("\n| Threads | Time (s) | Speedup | Efficiency |")
    print("|---------|----------|---------|------------|")
    for i, t in enumerate(threads_list):
        print(f"| {t} | {times[i]:.6f} | {speedups[i]:.2f}x | {efficiency[i]*100:.2f}% |")

if __name__ == "__main__":
    main()
