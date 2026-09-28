import numpy as np
from scipy.sparse import csc_matrix

indptr  = [0, 2, 4, 5, 8]
indices = [0, 1, 1, 2, 2, 0, 2, 3]
data    = [10, 3, 9, 7, 8, 2, 7, 5]

A_csc = csc_matrix((data, indices, indptr), shape=(4, 4))

print("Data:    ", A_csc.data)
print("Indices: ", A_csc.indices)
print("Indptr:  ", A_csc.indptr)

print("\nDense Matrix:\n", A_csc.toarray())