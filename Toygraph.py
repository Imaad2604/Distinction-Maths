import numpy as np
import scipy.sparse as sp

def pagerank(A, alpha=0.85, tol=1e-10, maxit=200):
    n = A.shape[0]
    A = sp.csr_matrix(A, dtype=float)
    out = np.asarray(A.sum(axis=1)).ravel()
    dangling = out == 0
    inv = np.where(dangling, 0.0, 1.0 / np.where(dangling, 1, out))
    PT = (sp.diags(inv) @ A).T.tocsr()
    x = np.full(n, 1.0 / n)
    for k in range(maxit):
        xn = alpha * (PT @ x + x[dangling].sum() / n) + (1 - alpha) / n
        if np.abs(xn - x).sum() < tol:
            return xn, k
        x = xn
    return x, maxit

rows = np.array([0, 0, 1, 2, 3, 3])
cols = np.array([1, 2, 2, 0, 0, 2])
data = np.ones(len(rows))
A = sp.coo_matrix((data, (rows, cols)), shape=(4, 4)).tocsr()

x, iters = pagerank(A)
print(f"PageRank ({iters} iters): {x}")
print(f"Sum: {x.sum():.10f}  (should be 1.0)")