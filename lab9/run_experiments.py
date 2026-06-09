#!/usr/bin/env python3

import argparse
import csv
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BIN = ROOT / "bin" / "matrix_transpose"
SUMMARY_RE = re.compile(
    r"matrix_size=(?P<matrix_size>\d+) kernel=(?P<kernel>\w+) "
    r"block_x=(?P<block_x>\d+) block_y=(?P<block_y>\d+) repeats=(?P<repeats>\d+) "
    r"time_ms=(?P<time_ms>[0-9.]+) bandwidth_gb_s=(?P<bandwidth_gb_s>[0-9.]+) "
    r"max_abs_error=(?P<max_abs_error>[0-9.eE+-]+)"
)


def parse_csv_ints(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_block(text):
    left, sep, right = text.lower().partition("x")
    if not sep:
        raise argparse.ArgumentTypeError("block sizes must use WIDTHxHEIGHT, e.g. 32x8")
    return int(left), int(right)


def run_command(cmd):
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(map(str, cmd))}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout


def parse_summary(stdout):
    match = SUMMARY_RE.search(stdout)
    if not match:
        raise RuntimeError(f"could not parse matrix_transpose output:\n{stdout}")
    row = match.groupdict()
    for key in ["matrix_size", "block_x", "block_y", "repeats"]:
        row[key] = int(row[key])
    for key in ["time_ms", "bandwidth_gb_s", "max_abs_error"]:
        row[key] = float(row[key])
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="512,1024,2048")
    parser.add_argument("--blocks", nargs="+", type=parse_block, default=[(16, 16), (32, 8), (32, 16)])
    parser.add_argument("--kernels", nargs="+", default=["naive", "tiled"])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    if not args.skip_build:
        run_command(["make", "all"])
    if not BIN.exists():
        raise FileNotFoundError(f"missing binary: {BIN}")

    rows = []
    for matrix_size in parse_csv_ints(args.sizes):
        for kernel in args.kernels:
            for block_x, block_y in args.blocks:
                cmd = [
                    str(BIN),
                    "--n",
                    str(matrix_size),
                    "--kernel",
                    kernel,
                    "--block-x",
                    str(block_x),
                    "--block-y",
                    str(block_y),
                    "--repeats",
                    str(args.repeats),
                    "--seed",
                    str(args.seed),
                ]
                row = parse_summary(run_command(cmd))
                rows.append(row)
                print(
                    f"N={matrix_size:4d} kernel={kernel:5s} block={block_x:2d}x{block_y:<2d} "
                    f"time={row['time_ms']:.6f} ms bandwidth={row['bandwidth_gb_s']:.3f} GB/s "
                    f"error={row['max_abs_error']:.1e}"
                )

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "transpose_summary.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "matrix_size",
            "kernel",
            "block_x",
            "block_y",
            "repeats",
            "time_ms",
            "bandwidth_gb_s",
            "max_abs_error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary saved to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
