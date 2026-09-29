#!/usr/bin/env python3
"""CSR Matrix-Vector Product Benchmark"""
import numpy as np
from scipy.sparse import csr_matrix
import time

def benchmark_csr():
    # Create a random sparse matrix
    n = 5000
    density = 0.001  # 0.1% sparsity
    nnz = int(n * n * density)
    rng = np.random.default_rng(42)
    rows = rng.integers(0, n, nnz)
    cols = rng.integers(0, n, nnz)
    data = rng.random(nnz)

    A = csr_matrix((data, (rows, cols)), shape=(n, n))

    # Warmup
    y_ref = A @ np.ones(n)

    # Time the operation
    start = time.perf_counter()
    y = A @ np.ones(n)
    elapsed = time.perf_counter() - start

    print(f"Matrix: {n} x {n}, nnz={nnz}, density={density*100:.2f}%")
    print(f"Time: {elapsed:.4f}s")
    print(f"Result shape: {y.shape}")
    print(f"Max difference from reference: {np.max(np.abs(y - y_ref))}")
    return elapsed

if __name__ == "__main__":
    benchmark_csr()
