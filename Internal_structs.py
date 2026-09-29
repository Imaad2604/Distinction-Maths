import numpy as np
import scipy.sparse as sp

row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
A_coo = sp.coo_matrix((data, (row, col)), shape=(3, 3))

print("--- COO Format ---")
print("Row indices:", A_coo.row)
print("Col indices:", A_coo.col)
print("Data:       ", A_coo.data)

A_csr = A_coo.tocsr()
print("\n--- CSR Format ---")
print("Data:       ", A_csr.data)
print("Indices:    ", A_csr.indices)
print("Indptr:     ", A_csr.indptr)

A_csc = A_coo.tocsc()
print("\n--- CSC Format ---")
print("Data:       ", A_csc.data)
print("Indices:    ", A_csc.indices)
print("Indptr:     ", A_csc.indptr)
