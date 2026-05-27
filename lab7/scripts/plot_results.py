import csv
from pathlib import Path
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent.parent
fig = root / "figures"
fig.mkdir(exist_ok=True)

# FFT plot
fft_rows = list(csv.DictReader((root / "data/fft_results.csv").open()))
procs = sorted({int(r["procs"]) for r in fft_rows})
Ns = sorted({int(r["N"]) for r in fft_rows})

plt.figure(figsize=(8, 5))
for p in procs:
    xs, ys = [], []
    for n in Ns:
        row = next(r for r in fft_rows if int(r["procs"]) == p and int(r["N"]) == n)
        xs.append(n)
        ys.append(float(row["total_time"]))
    plt.plot(xs, ys, marker="o", label=f"np={p}")
plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("N (FFT length)")
plt.ylabel("Total Time (s)")
plt.title("MPI FFT Performance")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(fig / "fft_time_vs_n.png", dpi=180)
plt.close()

# Heated plate runtime plot
hp_rows = list(csv.DictReader((root / "data/heated_plate_results.csv").open()))
ths = sorted({int(r["threads"]) for r in hp_rows})
hNs = sorted({int(r["N"]) for r in hp_rows})

plt.figure(figsize=(8, 5))
for t in ths:
    xs, ys = [], []
    for n in hNs:
        row = next(r for r in hp_rows if int(r["threads"]) == t and int(r["N"]) == n)
        xs.append(n)
        ys.append(float(row["wallclock"]))
    plt.plot(xs, ys, marker="o", label=f"threads={t}")
plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("N (grid size N x N)")
plt.ylabel("Wallclock (s)")
plt.title("parallel_for Heated Plate Runtime")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(fig / "heated_plate_time_vs_n.png", dpi=180)
plt.close()

# Memory plot (time -l RSS)
mem_rows = list(csv.DictReader((root / "data/heated_plate_memory_time_l.csv").open()))
xs = [int(r["threads"]) for r in mem_rows]
ys = [int(r["max_rss_kb"]) for r in mem_rows]

plt.figure(figsize=(7, 4.5))
plt.bar(xs, ys)
plt.xlabel("Threads")
plt.ylabel("Max RSS (KB) from /usr/bin/time -l")
plt.title("Memory Consumption (macOS substitute for massif)")
plt.tight_layout()
plt.savefig(fig / "heated_plate_memory_rss.png", dpi=180)
plt.close()

print("plots generated")
