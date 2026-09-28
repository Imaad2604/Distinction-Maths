import numpy as np
from scipy.sparse import lil_matrix

A_lil = lil_matrix((4, 4))

A_lil[0, 0] = 10
A_lil[0, 3] = 2
A_lil[1, 0] = 3
A_lil[1, 1] = 9
A_lil[2, 1] = 7
A_lil[2, 2] = 8
A_lil[2, 3] = 7
A_lil[3, 3] = 5

print("Rows:", A_lil.rows)
print("Data:", A_lil.data)


A_dense = np.array([
    [10, 0, 0, 2],
    [3, 9, 0, 0],
    [0, 7, 8, 7],
    [0, 0, 0, 5]
])

A_lil = lil_matrix(A_dense)

print("Rows:", A_lil.rows)
print("Data:", A_lil.data)