#include <iostream>
#include <pthread.h>
#include <stdlib.h>
#include <chrono>

using namespace std;

long long total_n;
long long total_m = 0;
int num_threads = 1;
pthread_mutex_t m_mutex = PTHREAD_MUTEX_INITIALIZER;

void* monte_carlo(void* arg) {
    long long thread_id = (long long)arg;
    long long local_n = total_n / num_threads;
    long long remainder = total_n % num_threads;
    if (thread_id < remainder) {
        local_n++;
    }

    long long local_m = 0;
    
    // Seed for random number generation
    unsigned int seed = chrono::system_clock::now().time_since_epoch().count() + thread_id;
    
    for (long long i = 0; i < local_n; ++i) {
        double x = (double)rand_r(&seed) / RAND_MAX;
        double y = (double)rand_r(&seed) / RAND_MAX;
        if (x * x + y * y <= 1.0) {
            local_m++;
        }
    }
    
    pthread_mutex_lock(&m_mutex);
    total_m += local_m;
    pthread_mutex_unlock(&m_mutex);
    
    return nullptr;
}

int main(int argc, char* argv[]) {
    if (argc >= 3) {
        total_n = atoll(argv[1]);
        num_threads = atoi(argv[2]);
    } else {
        cout << "Usage: " << argv[0] << " <n> <num_threads>\n";
        cout << "Falling back to stdin...\n";
        cout << "Input integer n [1024, 65536]: ";
        if (!(cin >> total_n)) {
            total_n = 65536;
        }
        cout << "Input num_threads: ";
        if (!(cin >> num_threads)) {
            num_threads = 4;
        }
    }

    auto start_time = chrono::high_resolution_clock::now();

    pthread_t* threads = new pthread_t[num_threads];
    for (long long i = 0; i < num_threads; ++i) {
        pthread_create(&threads[i], nullptr, monte_carlo, (void*)i);
    }

    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    double pi_estimate = 4.0 * total_m / (double)total_n;

    cout << "Total points n: " << total_n << endl;
    cout << "Points m in target: " << total_m << endl;
    cout << "Estimated pi: " << pi_estimate << endl;
    cout << "Time consumed t: " << elapsed.count() << " s" << endl;

    delete[] threads;
    return 0;
}
