import numpy as np
import scipy.sparse as sp

n = 1_000_000
density = 1e-6
nnz = int(n * n * density)          
rng = np.random.default_rng(0)

rows = rng.integers(0, n, nnz)
cols = rng.integers(0, n, nnz)
data = rng.random(nnz)

A = sp.coo_matrix((data, (rows, cols)), shape=(n, n))

def get_memory(M):
    return M.data.nbytes + M.indices.nbytes + M.indptr.nbytes

A_csr = A.tocsr()
A_csc = A.tocsc()

print(f"Matrix size: {n}x{n}, nnz: {A.nnz}")
print(f"COO memory: {A.data.nbytes + A.row.nbytes + A.col.nbytes:>15,} bytes")
print(f"CSR memory: {get_memory(A_csr):>15,} bytes")
print(f"CSC memory: {get_memory(A_csc):>15,} bytes")
print(f"Dense memory: {n * n * 8:,} bytes (approx {n*n*8 / 1e12:.1f} TB)")