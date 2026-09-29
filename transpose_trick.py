import time
import numpy as np
import scipy.sparse as sp

def bench(fn, reps=10):
    fn() 
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return np.median(ts)


def main():
    n = 500_000
    density = 1e-5
    rng = np.random.default_rng(0)

    nnz = int(n * n * density)
    rows = rng.integers(0, n, nnz)
    cols = rng.integers(0, n, nnz)
    data = rng.random(nnz)

    A_csr = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    print(f"n = {n}, nnz = {A_csr.nnz}")
    print("-" * 60)

    t_transpose = bench(lambda: A_csr.transpose())
    print(f"A_csr.transpose()           : {t_transpose*1000:.3f} ms  (O(nnz))")

    t_tocsc = bench(lambda: A_csr.tocsc())
    print(f"A_csr.tocsc()               : {t_tocsc*1000:.3f} ms  (O(nnz))")

    def o1_transpose_view():
        return sp.csc_matrix(
            (A_csr.data, A_csr.indices, A_csr.indptr),
            shape=(A_csr.shape[1], A_csr.shape[0]),
            copy=False,       
        )

    t_o1 = bench(o1_transpose_view)
    print(f"O(1) reinterpret as CSC(Aᵀ) : {t_o1*1000:.6f} ms  (O(1))")

    B = o1_transpose_view()
    x = rng.random(n)
    y1 = B @ x                                
    y2 = A_csr.T @ x                          
    print("Correct?", np.allclose(y1, y2))


if __name__ == "__main__":
    main()