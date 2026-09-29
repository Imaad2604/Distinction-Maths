import scipy.sparse as sp, scipy.sparse.linalg as spla, numpy as np
N = 100
T = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(N, N), dtype=float)
I = sp.identity(N)
A = (sp.kron(I, T) + sp.kron(T, I)).tocsr()   
b = np.ones(N * N)
u_direct = spla.spsolve(A.tocsc(), b)
u_cg, info = spla.cg(A, b)                     


print("A shape: ", A.shape)
print("A nnz:   ", A.nnz)
print("info:    ", info)
print("u_direct[:5]:", u_direct[:5])
print("u_cg[:5]:    ", u_cg[:5])
print("match:       ", np.allclose(u_direct, u_cg, atol=1e-6))
print("u.max():     ", u_cg.max())
print("u.min():     ", u_cg.min())