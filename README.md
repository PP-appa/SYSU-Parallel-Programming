# Parallel Programming and Algorithms

**Author:** Yanwei Lei  
**Institution:** Sun Yat-sen University (SYSU) - School of Computer Science and Engineering  
**Academic Year:** 2025/2026

This repository hosts a collection of performance-oriented experiments developed for the **Parallel Programming and Algorithms** course. It aims to demonstrate various synchronization methods, parallel programming models, and performance tuning architectures such as Cache optimization on multiprocessor systems. 

## Project Structure

The repository is modularly organized into several academic lab assignments:

### `lab0` - Basic Environment and Cache Optimizations
Focused on evaluating computational loops by exploiting locality and CPU caching algorithms to boost basic matrix multiplication.
- Matrix optimization strategies (IKJ vs IJK forms)
- MKL Math library integration checks
- Loop Unrolling (`gemm_unroll.cpp`)

### `lab1` - Introduction to Distributed Memory (MPI P2P)
Introduces the MPI framework operating on Message Passing paradigms. Built upon pure Point-to-Point blocking/non-blocking communication paths to evaluate load distribution for array calculations.

### `lab2` - MPI Collective Communication
Replaces primitive P2P messaging with sophisticated Broadcasts (`MPI_Bcast`) and Gather (`MPI_Gather`) pipelines to build a highly robust network scale-out model for processing giant matrices simultaneously over simulated clustered nodes.

### `lab3` - Shared Memory Multi-Threading (Pthreads)
Transitions from network passing to POSIX direct thread bindings operating inside memory limits of a singular physical die. Features implementations analyzing:
- Block data distribution models (`pthread_gemm.cpp`)
- Thread-safe Accumulators via local reductions (`pthread_sum.cpp`)
- An in-depth performance analysis investigating L1 Cache False Sharing penalties compared across standard compiler optimization layers.

## Development Stack

* OS Environment: Linux / Windows Subsystem for Linux (WSL)
* Compilers: GCC (`g++`)
* Dependencies: 
  * OpenMPI (for labs 1 and 2)
  * Pthreads (for lab 3 OS threading hooks)
* Recommended Optimizations: `-O3` (unless running architectural experiments like False Sharing simulations)

*Please review each individual `lab{N}/report.md` for granular mathematical proofs, algorithmic workflows, Amdahl's Law implications, and benchmark data tables regarding respective experiments.*
