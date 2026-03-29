import sys
import time
import random

def matrix_multiply(A, B, m, n, k):
    # Initialize result matrix C with zeros
    C = [[0.0 for _ in range(n)] for _ in range(m)]
    
    # Standard i-j-k loop
    for i in range(m):
        for j in range(n):
            for inner_k in range(k):
                C[i][j] += A[i][inner_k] * B[inner_k][j]
    return C

def save_matrix(filename, mat):
    with open(filename, 'w') as f:
        for row in mat:
            f.write(' '.join([f"{x:.4f}" for x in row]) + '\n')
def main():
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <m> <n> <k>")
        sys.exit(1)
        
    m = int(sys.argv[1])
    n = int(sys.argv[2])
    k = int(sys.argv[3])
    
    # Initialize A and B with random values
    A = [[random.random() for _ in range(k)] for _ in range(m)]
    B = [[random.random() for _ in range(n)] for _ in range(k)]
    
    start_time = time.time()
    
    C = matrix_multiply(A, B, m, n, k)
    
    end_time = time.time()
    
    print(f"Matrix size: m={m}, n={n}, k={k}")    save_matrix("A_py.txt", A)
    save_matrix("B_py.txt", B)
    save_matrix("C_py.txt", C)

    print(f"Execution time: {end_time - start_time:.6f} seconds")

if __name__ == "__main__":
    main()
