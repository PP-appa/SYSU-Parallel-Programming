#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mkl.h>

void save_matrix(const char* filename, double* mat, int rows, int cols) {
    FILE *f = fopen(filename, "w");
    if(!f) return;
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) fprintf(f, "%.4f ", mat[i*cols+j]);
        fprintf(f, "\n");
    }
    fclose(f);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <m> <n> <k>\n", argv[0]);
        return 1;
    }
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);

    double *A = (double*)mkl_malloc(m * k * sizeof(double), 64);
    double *B = (double*)mkl_malloc(k * n * sizeof(double), 64);
    double *C = (double*)mkl_malloc(m * n * sizeof(double), 64);

    for (int i = 0; i < m * k; i++) A[i] = (double)rand() / RAND_MAX;
    for (int i = 0; i < k * n; i++) B[i] = (double)rand() / RAND_MAX;

    struct timeval start, end;
    gettimeofday(&start, NULL);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1.0, A, k, B, n, 0.0, C, n);

    gettimeofday(&end, NULL);
    double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    
    printf("Matrix size: m=%d, n=%d, k=%d\n", m, n, k);
    printf("Execution time: %f seconds\n", time_spent);

    save_matrix("A.txt", A, m, k);
    save_matrix("B.txt", B, k, n);
    save_matrix("C.txt", C, m, n);

    mkl_free(A);
    mkl_free(B);
    mkl_free(C);
    return 0;
}
