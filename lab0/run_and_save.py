import os
import glob
import subprocess
import json
import re

def patch_cpp():
    for fpath in glob.glob('src/*.cpp'):
        with open(fpath, 'r') as f:
            code = f.read()
        if 'save_matrix' not in code:
            func = '''void save_matrix(const char* filename, double* mat, int rows, int cols) {
    FILE *f = fopen(filename, "w");
    if(!f) return;
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) fprintf(f, "%.4f ", mat[i*cols+j]);
        fprintf(f, "\\n");
    }
    fclose(f);
}\n'''
            code = code.replace('int main(', func + 'int main(')
            calls = '\n    save_matrix("A.txt", A, m, k);\n    save_matrix("B.txt", B, k, n);\n    save_matrix("C.txt", C, m, n);\n'
            code = code.replace('free(A);', calls + '    free(A);')
            code = code.replace('mkl_free(A);', calls + '    mkl_free(A);')
            with open(fpath, 'w') as f:
                f.write(code)

def patch_py():
    py_path = 'src/gemm_basic.py'
    with open(py_path, 'r') as f:
        py_code = f.read()
    if 'save_matrix' not in py_code:
        py_func = '''def save_matrix(filename, mat):
    with open(filename, 'w') as f:
        for row in mat:
            f.write(' '.join([f"{x:.4f}" for x in row]) + '\\n')\n'''
        py_code = py_code.replace('def main():', py_func + 'def main():')
        py_calls = '    save_matrix("A_py.txt", A)\n    save_matrix("B_py.txt", B)\n    save_matrix("C_py.txt", C)\n'
        py_code = re.sub(r'(\s*print\(f"Execution time)', py_calls + r'\1', py_code)
        with open(py_path, 'w') as f:
            f.write(py_code)

def build():
    os.makedirs('bin', exist_ok=True)
    subprocess.run('gcc -o bin/gemm_basic src/gemm_basic.cpp', shell=True)
    subprocess.run('gcc -o bin/gemm_ikj src/gemm_ikj.cpp', shell=True)
    subprocess.run('gcc -O3 -o bin/gemm_unroll src/gemm_unroll.cpp', shell=True)
    subprocess.run('gcc -O3 -o bin/gemm_mkl src/gemm_mkl.cpp -I/usr/include/mkl -lmkl_rt', shell=True)
    subprocess.run('gcc -O3 -o bin/gemm_O3 src/gemm_basic.cpp', shell=True)

def run_tests():
    results = {}
    commands = {
        "C++_Basic": "./bin/gemm_basic 1024 1024 1024",
        "C++_O3": "./bin/gemm_O3 1024 1024 1024",
        "C++_ikj": "./bin/gemm_ikj 1024 1024 1024",
        "C++_Unroll_O3": "./bin/gemm_unroll 1024 1024 1024",
        "C++_MKL": "./bin/gemm_mkl 1024 1024 1024",
        "Python_Basic": "python3 src/gemm_basic.py 1024 1024 1024"
    }
    for name, cmd in commands.items():
        print(f"Running {name}...")
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        m = re.search(r"Execution time:\s*([0-9\.]+)", res.stdout)
        if m:
            results[name] = float(m.group(1))
            print(f" -> {results[name]}s")
        else:
            results[name] = "Failed"
            print(" -> Failed")
            
    with open('times.json', 'w') as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    patch_cpp()
    patch_py()
    build()
    run_tests()
