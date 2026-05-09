#include <iostream>
#include <pthread.h>
#include <cmath>
#include <chrono>

using namespace std;

double a, b, c;
double delta_val;
double x1_val, x2_val;
bool delta_ready = false;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond_delta = PTHREAD_COND_INITIALIZER;

void* calc_delta(void* arg) {
    double d = b * b - 4 * a * c;
    
    pthread_mutex_lock(&mutex);
    delta_val = d;
    delta_ready = true;
    pthread_cond_broadcast(&cond_delta);
    pthread_mutex_unlock(&mutex);
    
    return nullptr;
}

void* calc_x1(void* arg) {
    pthread_mutex_lock(&mutex);
    while (!delta_ready) {
        pthread_cond_wait(&cond_delta, &mutex);
    }
    double d = delta_val;
    pthread_mutex_unlock(&mutex);
    
    if (d >= 0) {
        x1_val = (-b + sqrt(d)) / (2 * a);
    }
    return nullptr;
}

void* calc_x2(void* arg) {
    pthread_mutex_lock(&mutex);
    while (!delta_ready) {
        pthread_cond_wait(&cond_delta, &mutex);
    }
    double d = delta_val;
    pthread_mutex_unlock(&mutex);
    
    if (d >= 0) {
        x2_val = (-b - sqrt(d)) / (2 * a);
    }
    return nullptr;
}

int main() {
    cout << "Input a, b, c [-100, 100]: ";
    // We try to read, but if it runs unattended, we give default
    if (!(cin >> a >> b >> c)) {
        a = 1.0; b = -3.0; c = 2.0;
        cout << "Using defaults: 1.0 -3.0 2.0\n";
    }

    auto start_time = chrono::high_resolution_clock::now();

    pthread_t t1, t2, t3;
    // Thread 1: compute delta
    pthread_create(&t1, nullptr, calc_delta, nullptr);
    // Thread 2: compute x1
    pthread_create(&t2, nullptr, calc_x1, nullptr);
    // Thread 3: compute x2
    pthread_create(&t3, nullptr, calc_x2, nullptr);

    pthread_join(t1, nullptr);
    pthread_join(t2, nullptr);
    pthread_join(t3, nullptr);

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    if (delta_val < 0) {
        cout << "No real roots." << endl;
    } else {
        cout << "Roots: x1 = " << x1_val << ", x2 = " << x2_val << endl;
    }
    cout << "Time consumed t: " << elapsed.count() << " s" << endl;

    return 0;
}
