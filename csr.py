import numpy as np
from scipy.sparse import csr_matrix

A = np.array([
    [10, 0, 0, 2],
    [3, 9, 0, 0],
    [0, 7, 8, 7],
    [0, 0, 0, 5]
])

A_csr = csr_matrix(A)
print("Data:", A_csr.data)
print("Indices:", A_csr.indices)
print("Indptr:", A_csr.indptr)