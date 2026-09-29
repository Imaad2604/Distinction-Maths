import time
import numpy as np
import scipy.sparse as sp

def bench(fn, reps=5):
    fn() 
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return np.median(ts)

n = 10_000
density = 0.001

rows = np.random.randint(0, n, int(n*n*density))
cols = np.random.randint(0, n, int(n*n*density))
data = np.random.rand(len(rows))

print("Benchmarking Construction (n=10,000, density=0.001)")
print("COO build: ", bench(lambda: sp.coo_matrix((data, (rows, cols)), shape=(n, n))))


A_coo = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
print("COO->LIL:  ", bench(lambda: A_coo.tolil()))
print("COO->DOK:  ", bench(lambda: A_coo.todok()))
print("COO->CSR:  ", bench(lambda: A_coo.tocsr()))