#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
make

for p in 1 2 4 8; do
  echo "===== np=${p} ====="
  mpirun -np "$p" ./bin/mpi_fft 4096 20
  echo
 done
