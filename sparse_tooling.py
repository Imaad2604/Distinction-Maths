import numpy as np
import scipy.sparse as sp

rows = np.array([0, 0, 1, 1, 2, 2, 2])    
cols = np.array([0, 2, 1, 3, 0, 3, 4])    
data = np.array([5, 3, 4, 2, 5, 4, 1])    
R = sp.csr_matrix((data, (rows, cols)), shape=(3, 5))

S = (R.T @ R).tocsr()

print("R shape: ", R.shape, "nnz:", R.nnz)
print("S shape: ", S.shape, "nnz:", S.nnz)
print(S.toarray())