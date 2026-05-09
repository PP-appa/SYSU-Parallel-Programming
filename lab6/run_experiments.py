#!/usr/bin/env python3

import subprocess
import json
import time
import os

def run_experiment(num_threads, epsilon=0.001):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"Running with {num_threads} thread(s)")
    print(f"{'='*60}")
    
    cmd = [f"./bin/heated_plate_pthread", str(num_threads), str(epsilon)]
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        elapsed = time.time() - start_time
        
        output = result.stdout
        print(output)
        
        # 提取关键信息
        lines = output.strip().split('\n')
        wall_time = None
        iterations = None
        
        for line in lines:
            if "Wallclock time" in line:
                wall_time = float(line.split('=')[1].strip())
            if "Error tolerance achieved" not in line and "Iteration" in line and num not in line[0]:
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        iterations = int(parts[0])
                except:
                    pass
        
        return {
            'threads': num_threads,
            'wall_time': wall_time,
            'iterations': iterations,
            'elapsed': elapsed
        }
    except subprocess.TimeoutExpired:
        print(f"ERROR: Timeout after 300 seconds")
        return {
            'threads': num_threads,
            'wall_time': None,
            'iterations': None,
            'elapsed': 300
        }
    except Exception as e:
        print(f"ERROR: {e}")
        return {
            'threads': num_threads,
            'wall_time': None,
            'iterations': None,
            'elapsed': None
        }

def main():
    # 检查可执行文件是否存在
    if not os.path.exists("./bin/heated_plate_pthread"):
        print("ERROR: Executable not found. Please compile first with: make all")
        return
    
    results = []
    
    # 测试不同的线程数
    for num_threads in [1, 2, 4, 8]:
        result = run_experiment(num_threads)
        results.append(result)
    
    # 打印总结
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Threads':<10} {'Time (s)':<15} {'Speedup':<10}")
    print(f"{'-'*60}")
    
    baseline_time = results[0]['wall_time']
    
    for result in results:
        threads = result['threads']
        wall_time = result['wall_time']
        if wall_time and baseline_time:
            speedup = baseline_time / wall_time
            print(f"{threads:<10} {wall_time:<15.3f} {speedup:<10.2f}x")
        else:
            print(f"{threads:<10} {'N/A':<15} {'N/A':<10}")
    
    # 保存结果到JSON
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to experiment_results.json")

if __name__ == "__main__":
    main()
