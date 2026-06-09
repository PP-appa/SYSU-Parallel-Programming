CUDA_HOME ?= /usr/local/cuda
NVCC_PATH := $(shell command -v nvcc 2>/dev/null || \
	if [ -x "/usr/local/cuda/bin/nvcc" ]; then printf "%s" "/usr/local/cuda/bin/nvcc"; \
	elif [ -x "/usr/local/cuda-12.6/bin/nvcc" ]; then printf "%s" "/usr/local/cuda-12.6/bin/nvcc"; \
	else printf "%s" "nvcc"; fi)
NVCC ?= $(NVCC_PATH)
CUDAFLAGS ?= -O3 -std=c++17 -Xcompiler -Wall,-Wextra

HELLO_SRC := src/cuda_hello.cu
TRANSPOSE_SRC := src/matrix_transpose.cu
HELLO_BIN := bin/cuda_hello
TRANSPOSE_BIN := bin/matrix_transpose

.PHONY: all test clean hello transpose experiments

all: $(HELLO_BIN) $(TRANSPOSE_BIN)

$(HELLO_BIN): $(HELLO_SRC)
	@mkdir -p bin
	$(NVCC) $(CUDAFLAGS) $< -o $@

$(TRANSPOSE_BIN): $(TRANSPOSE_SRC)
	@mkdir -p bin
	$(NVCC) $(CUDAFLAGS) $< -o $@

hello: $(HELLO_BIN)
	./$(HELLO_BIN) 2 4 4

transpose: $(TRANSPOSE_BIN)
	./$(TRANSPOSE_BIN) --n 1024 --kernel tiled --block-x 32 --block-y 8

experiments: $(TRANSPOSE_BIN)
	python3 run_experiments.py

test:
	python3 tests/test_lab9_static.py

clean:
	rm -f $(HELLO_BIN) $(TRANSPOSE_BIN)
