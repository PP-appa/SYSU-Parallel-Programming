#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

make

FFT_CSV=data/fft_results.csv
HP_CSV=data/heated_plate_results.csv
MEM_CSV=data/heated_plate_memory_time_l.csv

mkdir -p data

echo "procs,N,nits,error,total_time,time_per_call,mflops" > "$FFT_CSV"
for p in 1 2 4 8; do
  for N in 1024 2048 4096 8192; do
    nits=10
    out=$(mpirun -np "$p" ./bin/mpi_fft "$N" "$nits")
    echo "$out" > "data/fft_np${p}_N${N}.log"
    err=$(echo "$out" | awk -F= '/error=/{print $2}')
    tt=$(echo "$out" | awk -F= '/total_time=/{print $2}' | awk '{print $1}')
    tpc=$(echo "$out" | awk -F= '/time_per_call=/{print $2}' | awk '{print $1}')
    mf=$(echo "$out" | awk -F= '/mflops=/{print $2}')
    echo "${p},${N},${nits},${err},${tt},${tpc},${mf}" >> "$FFT_CSV"
  done
done

echo "N,threads,epsilon,iterations,final_diff,wallclock" > "$HP_CSV"
for N in 8 16 32 64 128 256; do
  for t in 1 2 4 8; do
    eps=0.001
    out=$(./bin/heated_plate_parallel_for "$N" "$t" "$eps")
    echo "$out" > "data/hp_N${N}_t${t}.log"
    it=$(echo "$out" | awk -F= '/iterations=/{print $2}')
    fd=$(echo "$out" | awk -F= '/final_diff=/{print $2}')
    wc=$(echo "$out" | awk -F= '/wallclock=/{print $2}')
    echo "${N},${t},${eps},${it},${fd},${wc}" >> "$HP_CSV"
  done
done

echo "N,threads,max_rss_kb,wallclock_s" > "$MEM_CSV"
for t in 1 2 4 8; do
  N=256
  memlog=$(mktemp)
  /usr/bin/time -l ./bin/heated_plate_parallel_for "$N" "$t" 0.001 >/tmp/hp_mem_${t}.out 2>"$memlog"
  rss=$(awk '/maximum resident set size/{print $1}' "$memlog")
  wct=$(awk '/real/{print $1}' "$memlog" | head -n 1)
  # Fallback wallclock from program output for portability
  if [ -z "${wct}" ]; then
    wct=$(awk -F= '/wallclock=/{print $2}' /tmp/hp_mem_${t}.out)
  fi
  echo "${N},${t},${rss},${wct}" >> "$MEM_CSV"
  cp "$memlog" "data/time_l_t${t}.log"
  rm -f "$memlog"
done

