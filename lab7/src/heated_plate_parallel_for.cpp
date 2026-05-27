#include <pthread.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct WorkData {
  int n;
  double* u;
  double* w;
  double* diff_ptr;
  pthread_mutex_t* diff_lock_ptr;
};

static inline double& at(double* a, int n, int i, int j) { return a[i * n + j]; }

void* copy_functor(int row, void* arg) {
  WorkData* d = static_cast<WorkData*>(arg);
  for (int j = 0; j < d->n; ++j) at(d->u, d->n, row, j) = at(d->w, d->n, row, j);
  return nullptr;
}

void* update_functor(int row, void* arg) {
  WorkData* d = static_cast<WorkData*>(arg);
  const int n = d->n;
  if (row > 0 && row < n - 1) {
    for (int j = 1; j < n - 1; ++j) {
      at(d->w, n, row, j) = (at(d->u, n, row - 1, j) + at(d->u, n, row + 1, j) +
                             at(d->u, n, row, j - 1) + at(d->u, n, row, j + 1)) /
                            4.0;
    }
  }
  return nullptr;
}

void* diff_functor(int row, void* arg) {
  WorkData* d = static_cast<WorkData*>(arg);
  const int n = d->n;
  if (row > 0 && row < n - 1) {
    double local_diff = 0.0;
    for (int j = 1; j < n - 1; ++j) {
      const double dv = std::fabs(at(d->w, n, row, j) - at(d->u, n, row, j));
      local_diff = std::max(local_diff, dv);
    }
    pthread_mutex_lock(d->diff_lock_ptr);
    *d->diff_ptr = std::max(*d->diff_ptr, local_diff);
    pthread_mutex_unlock(d->diff_lock_ptr);
  }
  return nullptr;
}

void parallel_for(int start, int end, int inc, void* (*functor)(int, void*), void* arg,
                  int n_threads) {
  if (n_threads <= 0 || inc <= 0 || start >= end) return;

  int total_iterations = (end - start + inc - 1) / inc;
  if (total_iterations == 0) return;
  if (n_threads > total_iterations) n_threads = total_iterations;

  std::vector<pthread_t> threads(n_threads);
  struct ThreadData {
    int start;
    int end;
    int inc;
    void* (*functor)(int, void*);
    void* arg;
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

    pthread_create(
        &threads[i], nullptr,
        [](void* p) -> void* {
          ThreadData* data = static_cast<ThreadData*>(p);
          for (int k = data->start; k < data->end; k += data->inc) data->functor(k, data->arg);
          return nullptr;
        },
        &t_data[i]);
    current_iter += iters_for_this_thread;
  }

  for (int i = 0; i < n_threads; ++i) pthread_join(threads[i], nullptr);
}

int main(int argc, char* argv[]) {
  int n = 500;
  int num_threads = 4;
  double epsilon = 1e-3;

  if (argc > 1) n = std::atoi(argv[1]);
  if (argc > 2) num_threads = std::atoi(argv[2]);
  if (argc > 3) epsilon = std::atof(argv[3]);

  if (n < 4 || num_threads <= 0 || epsilon <= 0) {
    std::fprintf(stderr, "Usage: %s N threads epsilon\n", argv[0]);
    return 1;
  }

  std::vector<double> u(n * n, 0.0), w(n * n, 0.0);
  pthread_mutex_t diff_lock = PTHREAD_MUTEX_INITIALIZER;
  double global_diff = 0.0;

  for (int i = 1; i < n - 1; ++i) {
    at(w.data(), n, i, 0) = 100.0;
    at(w.data(), n, i, n - 1) = 100.0;
  }
  for (int j = 0; j < n; ++j) {
    at(w.data(), n, 0, j) = 0.0;
    at(w.data(), n, n - 1, j) = 100.0;
  }

  double mean = 0.0;
  for (int i = 1; i < n - 1; ++i) mean += at(w.data(), n, i, 0) + at(w.data(), n, i, n - 1);
  for (int j = 0; j < n; ++j) mean += at(w.data(), n, n - 1, j) + at(w.data(), n, 0, j);
  mean /= static_cast<double>(2 * n + 2 * n - 4);

  for (int i = 1; i < n - 1; ++i)
    for (int j = 1; j < n - 1; ++j) at(w.data(), n, i, j) = mean;

  WorkData wd{n, u.data(), w.data(), &global_diff, &diff_lock};

  int iters = 0;
  double diff = epsilon;
  auto t1 = std::chrono::high_resolution_clock::now();
  while (diff >= epsilon) {
    parallel_for(0, n, 1, copy_functor, &wd, num_threads);
    parallel_for(1, n - 1, 1, update_functor, &wd, num_threads);
    global_diff = 0.0;
    parallel_for(1, n - 1, 1, diff_functor, &wd, num_threads);
    diff = global_diff;
    ++iters;
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  const double sec = std::chrono::duration<double>(t2 - t1).count();

  std::printf("HEATED_PLATE_PARALLEL_FOR\n");
  std::printf("N=%d threads=%d epsilon=%g\n", n, num_threads, epsilon);
  std::printf("iterations=%d\n", iters);
  std::printf("final_diff=%.9f\n", diff);
  std::printf("wallclock=%.9f\n", sec);
  return 0;
}
