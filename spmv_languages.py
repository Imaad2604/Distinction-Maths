import os
import sys
import time
import shutil
import subprocess
import platform
import numpy as np
import scipy
import scipy.sparse as sp


def ensure_cpp_binary():
    src = os.path.join(os.path.dirname(__file__), "csr_bench.cpp")
    binary = os.path.join(os.path.dirname(__file__), "csr_bench")
    if not os.path.exists(binary) or os.path.getmtime(binary) < os.path.getmtime(src):
        compiler = shutil.which("g++") or shutil.which("clang++")
        if compiler is None:
            sys.exit("No g++ or clang++ found. Install one and rerun.")
        print(f"Compiling with {compiler} -O3 ...")
        subprocess.run(
            [compiler, "-O3", "-std=c++17", src, "-o", binary],
            check=True,
        )
    return binary


def python_spmv(indptr, indices, data, x, n_rows):
    y = np.zeros(n_rows)
    for i in range(n_rows):
        s = 0.0
        for k in range(indptr[i], indptr[i + 1]):
            s += data[k] * x[indices[k]]
        y[i] = s
    return y


def cpp_spmv(binary, A_csr, x):
    n_rows, n_cols = A_csr.shape
    nnz = A_csr.nnz
    lines = []
    lines.append(f"{n_rows} {n_cols} {nnz}")
    lines.append(" ".join(map(str, A_csr.indptr)))
    lines.append(" ".join(map(str, A_csr.indices)))
    lines.append(" ".join(f"{v:.17g}" for v in A_csr.data))
    lines.append(" ".join(f"{v:.17g}" for v in x))
    stdin = "\n".join(lines) + "\n"

    out = subprocess.run(
        [binary], input=stdin, capture_output=True, text=True, check=True,
    ).stdout

    t_ms = None
    checksum = None
    for line in out.splitlines():
        if line.startswith("CSR_CPP_TIME_MS"):
            t_ms = float(line.split()[1])
        elif line.startswith("CSR_CPP_CHECKSUM"):
            checksum = float(line.split()[1])
    return t_ms, checksum


def bench(fn, reps=10):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    print(f"Machine  : {platform.node()} ({platform.machine()})")
    print(f"OS       : {platform.system()} {platform.release()}")
    print(f"Python   : {platform.python_version()}")
    print(f"NumPy    : {np.__version__}")
    print(f"SciPy    : {scipy.__version__}")
    print("-" * 68)

    binary = ensure_cpp_binary()

    n = 200_000
    density = 5e-5
    nnz = int(n * n * density)
    rng = np.random.default_rng(0)
    rows = rng.integers(0, n, nnz)
    cols = rng.integers(0, n, nnz)
    data = rng.random(nnz)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    A.sum_duplicates()

    x = rng.random(n)

    print(f"Matrix   : n = {n}, nnz = {A.nnz}")
    print(f"Density  : {A.nnz / (n * n):.2e}")
    print("-" * 68)

    y_ref = A @ x

    t_py = bench(lambda: python_spmv(A.indptr, A.indices, A.data, x, n), reps=3)
    y_py = python_spmv(A.indptr, A.indices, A.data, x, n)
    ok_py = np.allclose(y_py, y_ref, atol=1e-8)

    t_cpp_ms, checksum = cpp_spmv(binary, A, x)
    t_cpp = t_cpp_ms / 1000.0  # ms -> s
    ok_cpp = np.isclose(checksum, y_ref.sum(), atol=1e-6)

    t_scipy = bench(lambda: A @ x, reps=10)
    y_sp = A @ x
    ok_sp = np.allclose(y_sp, y_ref, atol=1e-12)

    print(f"Correct vs SciPy -> pure Python: {ok_py}, C++: {ok_cpp}")
    print()
    print(f"{'impl':<16}{'time (s)':>14}{'GFLOP/s':>12}{'speed-up':>12}")
    print("-" * 54)

    flops = 2.0 * A.nnz
    for name, t in [("pure Python", t_py), ("C++ CSR", t_cpp), ("SciPy CSR", t_scipy)]:
        gflops = flops / t / 1e9
        speedup = t_py / t
        print(f"{name:<16}{t:>14.6f}{gflops:>12.4f}{speedup:>11.1f}x")

    print("-" * 54)
    print("(speed-up column is relative to pure Python)")


if __name__ == "__main__":
    main()
