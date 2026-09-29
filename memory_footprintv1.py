import time
import numpy as np
import scipy.sparse as sp


def bench(fn, reps=7):
    fn()  
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return np.median(ts)


def main():
    n, d = 100_000, 1e-4
    rng = np.random.default_rng(0)

    nnz = int(n * n * d)
    rows = rng.integers(0, n, nnz)
    cols = rng.integers(0, n, nnz)
    data = rng.random(nnz)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n))

    x = rng.random(n)

    print(f"n = {n}, requested nnz = {nnz}, actual nnz = {A.nnz}")
    print(f"{'fmt':<5} {'convert (s)':>14} {'matvec (s)':>14} {'GFLOP/s':>10}")
    print("-" * 50)

    for fmt in ["coo", "csr", "csc"]:
        M = A.asformat(fmt)
        t_convert = bench(lambda: A.asformat(fmt))
        t_matvec = bench(lambda: M @ x)
        gflops = (2 * M.nnz) / t_matvec / 1e9
        print(f"{fmt:<5} {t_convert:>14.6f} {t_matvec:>14.6f} {gflops:>10.3f}")


if __name__ == "__main__":
    main()