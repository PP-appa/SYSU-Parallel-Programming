#ifndef PARALLEL_FOR_H
#define PARALLEL_FOR_H

#ifdef __cplusplus
extern "C" {
#endif

// 定义 parallel_for 函数的接口
// start: 循环开始索引 (包含)
// end: 循环结束索引 (不包含)
// inc: 索引自增量
// functor: 被执行的函数指针，接受两个参数：当前索引 idx 和自定义数据块指针 arg
// arg: 传递给 functor 的自定义数据指针
// num_threads: 启动的线程数
void parallel_for(int start, int end, int inc, 
                  void *(*functor)(int, void*), 
                  void *arg, int num_threads);

// 为了支持题目说明最后"传递 for_index"的内部实现，我们也提供一个基于块调用的 parallel_for_block
// 这允许用户自己在 functor 内部写 for 循环，从而减少高频调用的开销
struct for_index {
    int start;
    int end;
    int increment;
};

void parallel_for_block(int start, int end, int inc, 
                        void *(*functor)(void*), 
                        void *arg, int num_threads);

#ifdef __cplusplus
}
#endif

#endif
