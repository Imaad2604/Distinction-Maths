import numpy as np 

np.random.seed(2)
A = np.random.rand(4,4)
print(f"A  :{A}")

A_inv = np.linalg.inv(A)
print(f"A inverse :{A_inv}")


print(np.matmul(A_inv , A))