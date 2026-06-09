#!/usr/bin/env python3

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


class Lab9StaticTests(unittest.TestCase):
    def test_expected_files_exist(self):
        expected = [
            "Makefile",
            "README.md",
            "run_experiments.py",
            "src/cuda_hello.cu",
            "src/matrix_transpose.cu",
        ]
        missing = [path for path in expected if not (ROOT / path).exists()]
        self.assertEqual([], missing, f"missing lab9 files: {missing}")

    def test_makefile_exposes_cuda_targets(self):
        makefile = read("Makefile")
        self.assertIn("NVCC", makefile)
        self.assertIn("bin/cuda_hello", makefile)
        self.assertIn("bin/matrix_transpose", makefile)
        self.assertIn("test:", makefile)

    def test_makefile_falls_back_to_local_cuda_nvcc(self):
        makefile = read("Makefile")
        self.assertIn("NVCC_PATH", makefile)
        self.assertIn("/usr/local/cuda/bin/nvcc", makefile)

    def test_cuda_hello_uses_required_grid_and_block_output(self):
        source = read("src/cuda_hello.cu")
        self.assertIn("hello_kernel<<<n, dim3(m, k)>>>", source)
        self.assertIn("Hello World from Thread (%d, %d) in Block %d!", source)
        self.assertIn("Hello World from the host!", source)

    def test_matrix_transpose_contains_naive_and_tiled_kernels(self):
        source = read("src/matrix_transpose.cu")
        self.assertIn("__global__ void transpose_naive", source)
        self.assertIn("__global__ void transpose_tiled", source)
        self.assertIn("__shared__", source)
        self.assertIn("cudaEventElapsedTime", source)
        self.assertIn("max_abs_error", source)

    def test_experiment_script_records_required_factors(self):
        script = read("run_experiments.py")
        for token in [
            "matrix_size",
            "kernel",
            "block_x",
            "block_y",
            "time_ms",
            "bandwidth_gb_s",
            "max_abs_error",
        ]:
            self.assertIn(token, script)


if __name__ == "__main__":
    unittest.main()
