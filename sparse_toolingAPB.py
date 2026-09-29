import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

rows = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5]
cols = [0, 1, 2, 0, 1, 3, 2, 4, 5, 3, 4, 0, 5, 1]
data = [5, 4, 3, 5, 4, 2, 4, 5, 3, 4, 5, 3, 4, 2]
R = sp.csr_matrix((data, (rows, cols)), shape=(6, 8))

k = 2
U, sigma, Vt = svds(R.astype(float), k=k)
print("U shape:", U.shape)
print("Sigma shape:", sigma.shape)
print("Vt shape:", Vt.shape)