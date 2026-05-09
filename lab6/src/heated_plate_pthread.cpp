#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <vector>
#include <pthread.h>

#define M 500
#define N 500

double u[M][N];
double w[M][N];
double epsilon = 0.001;
int num_threads = 4;
double global_diff = 0.0;
pthread_mutex_t diff_lock = PTHREAD_MUTEX_INITIALIZER;

struct WorkData {
    double *diff_ptr;
    pthread_mutex_t *diff_lock_ptr;
};

void* copy_functor(int row, void* arg) {
    for (int j = 0; j < N; j++) {
        u[row][j] = w[row][j];
    }
    return nullptr;
}

void* update_functor(int row, void* arg) {
    if (row > 0 && row < M - 1) {
        for (int j = 1; j < N - 1; j++) {
            w[row][j] = (u[row-1][j] + u[row+1][j] + u[row][j-1] + u[row][j+1]) / 4.0;
        }
    }
    return nullptr;
}

void* diff_functor(int row, void* arg) {
    WorkData* data = (WorkData*)arg;
    if (row > 0 && row < M - 1) {
        double local_diff = 0.0;
        for (int j = 1; j < N - 1; j++) {
            double diff_val = fabs(w[row][j] - u[row][j]);
            if (local_diff < diff_val) {
                local_diff = diff_val;
            }
        }
        pthread_mutex_lock(data->diff_lock_ptr);
        if (*data->diff_ptr < local_diff) {
            *data->diff_ptr = local_diff;
        }
        pthread_mutex_unlock(data->diff_lock_ptr);
    }
    return nullptr;
}

void parallel_for(int start, int end, int inc,
                  void *(*functor)(int, void*),
                  void *arg, int n_threads) {
    if (n_threads <= 0 || inc <= 0 || start >= end) return;

    int total_iterations = (end - start + inc - 1) / inc;
    if (total_iterations == 0) return;

    if (n_threads > total_iterations) {
        n_threads = total_iterations;
    }

    std::vector<pthread_t> threads(n_threads);
    
    struct ThreadData {
        int start;
        int end;
        int inc;
        void *(*functor)(int, void*);
        void *arg;
    };
    std::vector<ThreadData> t_data(n_threads);

    int iter_per_thread = total_iterations / n_threads;
    int remainder = total_iterations % n_threads;
    int current_iter = 0;

    for (int i = 0; i < n_threads; ++i) {
        int iters_for_this_thread = iter_per_thread + (i < remainder ? 1 : 0);
        
        t_data[i].start = start + current_iter * inc;
        t_data[i].end = t_data[i].start + iters_for_this_thread * inc;
        t_data[i].inc = inc;
        t_data[i].functor = functor;
        t_data[i].arg = arg;

        if (t_data[i].end > end) t_data[i].end = end;

        pthread_create(&threads[i], nullptr, 
                      [](void* p) -> void* {
                          ThreadData* data = (ThreadData*)p;
                          for (int i = data->start; i < data->end; i += data->inc) {
                              data->functor(i, data->arg);
                          }
                          return nullptr;
                      }, &t_data[i]);

        current_iter += iters_for_this_thread;
    }

    for (int i = 0; i < n_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }
}

int main(int argc, char* argv[]) {
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }
    if (argc > 2) {
        epsilon = atof(argv[2]);
    }

    printf("\n");
    printf("HEATED_PLATE_PTHREAD\n");
    printf("  A program to solve for the steady state temperature distribution\n");
    printf("  over a rectangular plate using Pthreads.\n");
    printf("\n");
    printf("  Spatial grid of %d by %d points.\n", M, N);
    printf("  The iteration will be repeated until the change is <= %e\n", epsilon);
    printf("  Number of threads = %d\n", num_threads);

    double mean = 0.0;

    for (int i = 1; i < M - 1; i++) {
        w[i][0] = 100.0;
        w[i][N-1] = 100.0;
    }

    for (int j = 0; j < N; j++) {
        w[0][j] = 0.0;
        w[M-1][j] = 100.0;
    }

    for (int i = 1; i < M - 1; i++) {
        mean += w[i][0] + w[i][N-1];
    }
    for (int j = 0; j < N; j++) {
        mean += w[M-1][j] + w[0][j];
    }

    mean = mean / (double)(2 * M + 2 * N - 4);
    printf("\n");
    printf("  MEAN = %f\n", mean);

    for (int i = 1; i < M - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            w[i][j] = mean;
        }
    }

    int iterations = 0;
    int iterations_print = 1;
    printf("\n");
    printf(" Iteration  Change\n");
    printf("\n");

    auto wtime_start = std::chrono::high_resolution_clock::now();
    double diff = epsilon;
    
    WorkData work_data;
    work_data.diff_ptr = &global_diff;
    work_data.diff_lock_ptr = &diff_lock;

    while (epsilon <= diff) {
        parallel_for(0, M, 1, copy_functor, nullptr, num_threads);
        parallel_for(1, M - 1, 1, update_functor, nullptr, num_threads);
        global_diff = 0.0;
        parallel_for(1, M - 1, 1, diff_functor, &work_data, num_threads);
        diff = global_diff;

        iterations++;
        if (iterations == iterations_print) {
            printf("  %8d  %f\n", iterations, diff);
            iterations_print = 2 * iterations_print;
        }
    }

    auto wtime_end = std::chrono::high_resolution_clock::now();
    double wtime = std::chrono::duration<double>(wtime_end - wtime_start).count();

    printf("\n");
    printf("  %8d  %f\n", iterations, diff);
    printf("\n");
    printf("  Error tolerance achieved.\n");
    printf("  Wallclock time = %f\n", wtime);
    printf("\n");
    printf("HEATED_PLATE_PTHREAD:\n");
    printf("  Normal end of execution.\n");

    return 0;
}
