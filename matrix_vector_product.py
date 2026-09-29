import time
import numpy as np
import scipy.sparse as sp

def bench(fn, reps=10):
    fn()  # warm-up
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return np.median(ts)

def main():
    n = 100_000
    density = 1e-4
    rng = np.random.default_rng(0)

    nnz = int(n * n * density)
    rows = rng.integers(0, n, nnz)
    cols = rng.integers(0, n, nnz)
    data = rng.random(nnz)

    A_coo = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    x = rng.random(n)

    print(f"Benchmarking Matvec (n={n}, requested nnz={nnz}, actual nnz={A_coo.nnz})")
    print("-" * 60)

    for fmt in ["coo", "csr", "csc", "lil", "dok"]:
        M = A_coo.asformat(fmt)
        t = bench(lambda: M @ x)
        gflops = (2 * M.nnz) / t / 1e9
        print(f"{fmt.upper():<4} matvec: {t:.5f}s | {gflops:.3f} GFLOP/s")

if __name__ == "__main__":
    main()