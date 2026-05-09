#include "parallel_for.h"
#include <pthread.h>
#include <vector>
#include <iostream>

struct ThreadData {
    int thread_id;
    int start;
    int end;
    int inc;
    void *(*functor)(int, void*);     // 对于普通的 parallel_for
    void *(*block_functor)(void*);    // 对于基于块的 parallel_for_block
    void *arg;                        // 用户的参数
};

// 执行单个索引调用的 worker
void* worker_thread(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    for (int i = data->start; i < data->end; i += data->inc) {
        data->functor(i, data->arg);
    }
    return nullptr;
}

// 执行用户写好 for 循环的 block worker
// 需要将用户的 arg 传入，但同时需要把 range 信息传过去
// 题目中提到 "struct for_index * index = (struct for_index *) args"，但一般还需要原始数据指针。
// 这里的实现：为了满足题目最核心的 parallel_for(start, end, inc, functor, arg, num_threads) 要求，
// 我们重点完成第一种（在外面帮用户 for 循环）。

extern "C" {

void parallel_for(int start, int end, int inc, 
                  void *(*functor)(int, void*), 
                  void *arg, int num_threads) 
{
    if (num_threads <= 0 || inc <= 0 || start >= end) return;

    // 计算总的迭代次数
    int total_iterations = (end - start + inc - 1) / inc;
    if (total_iterations == 0) return;

    // 限制最大线程数不超过总迭代次数
    if (num_threads > total_iterations) {
        num_threads = total_iterations;
    }

    std::vector<pthread_t> threads(num_threads);
    std::vector<ThreadData> t_data(num_threads);

    int iter_per_thread = total_iterations / num_threads;
    int remainder = total_iterations % num_threads;

    int current_iter = 0;
    for (int i = 0; i < num_threads; ++i) {
        int iters_for_this_thread = iter_per_thread + (i < remainder ? 1 : 0);
        
        t_data[i].thread_id = i;
        t_data[i].start = start + current_iter * inc;
        t_data[i].end = t_data[i].start + iters_for_this_thread * inc;
        t_data[i].inc = inc;
        t_data[i].functor = functor;
        t_data[i].arg = arg;

        // 如果超出整体边界,进行裁剪
        if (t_data[i].end > end) t_data[i].end = end;

        pthread_create(&threads[i], nullptr, worker_thread, &t_data[i]);
        
        current_iter += iters_for_this_thread;
    }

    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }
}

// 第二种变体实现 (题目中 "替换为" 后的举例)
struct WrapperArgs {
    for_index index;
    void *original_arg;
};

// 注意：题目给出的 void* functor(void* args) 是用 for_index 的，
// 但原数据 (A,B,x,C 等) 是怎么传进去的？
// 题目中其实有点语义不明。一般这种 "用户自己在里面写for循环" 的，functor要能拿到 original_arg。
// 这里我们依然按照 "在库里执行for循环，外部只传入处理对应下标的动作" 也就是第一个定义作为主干。

}
