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
density = 0.01
A_coo = sp.random(n, n, density=density, format="coo", random_state=0)
A_csr = A_coo.tocsr()
A_lil = A_coo.tolil()
A_dok = A_coo.todok()

# Access 1000 random elements
indices = np.random.randint(0, n, (1000, 2))

def access_all(M):
    for i, j in indices:
        _ = M[i, j]

print("Element access (1000 random accesses):")
print("CSR: ", bench(lambda: access_all(A_csr)))
print("LIL: ", bench(lambda: access_all(A_lil)))
print("DOK: ", bench(lambda: access_all(A_dok)))