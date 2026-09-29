import numpy as np
import scipy.sparse as sp

n = 1000
A_coo = sp.random(n, n, density=0.01, format="coo", random_state=42)
x = np.random.rand(n)

y_scipy = A_coo.tocsr() @ x

def my_spmv(coo_matrix, x_vector):
    y = np.zeros(coo_matrix.shape[0])
    for i, j, v in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
        y[i] += v * x_vector[j]
    return y

y_mine = my_spmv(A_coo, x)

print("Do they match?", np.allclose(y_mine, y_scipy))