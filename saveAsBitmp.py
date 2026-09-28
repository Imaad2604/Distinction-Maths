import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def build_problem(n):
    """
    Builds the sparse matrix A and right-hand side vector b for a 2D grid.
    This is a placeholder implementing a 2D Laplacian (5-point stencil).
    """
    m = n * n
    # We will use COO format (similar to Eigen's Triplet)
    rows, cols, data = [], [], []
    
    # Build the 2D Laplacian matrix
    for i in range(n):
        for j in range(n):
            k = i * n + j
            # Diagonal element
            rows.append(k)
            cols.append(k)
            data.append(4.0)
            
            # Off-diagonals (neighbors)
            if i > 0: # Top neighbor
                rows.append(k)
                cols.append(k - n)
                data.append(-1.0)
            if i < n - 1: # Bottom neighbor
                rows.append(k)
                cols.append(k + n)
                data.append(-1.0)
            if j > 0: # Left neighbor
                rows.append(k)
                cols.append(k - 1)
                data.append(-1.0)
            if j < n - 1: # Right neighbor
                rows.append(k)
                cols.append(k + 1)
                data.append(-1.0)

    # Create the sparse matrix in CSR format (efficient for solving)
    A = sp.csr_matrix((data, (rows, cols)), shape=(m, m))
    
    # Right-hand side vector b (e.g., all ones, or a sinusoidal source)
    b = np.ones(m)
    
    return A, b

def save_as_bitmap(x, n, filename):
    """
    Saves the solution vector x as an image.
    """
    # Reshape the 1D vector into a 2D image
    image = x.reshape((n, n))
    
    # Normalize to [0, 1] for image saving
    image = (image - image.min()) / (image.max() - image.min())
    
    # Save using Matplotlib (or you could use PIL/cv2)
    plt.imsave(filename, image, cmap='gray')
    print(f"Image saved to {filename}")

def main():
    if len(sys.argv) != 2:
        print("Error: expected one and only one argument.")
        sys.exit(-1)

    n = 300    # size of the image
    m = n * n  # number of unknowns (=number of pixels)

    # Assembly
    A, b = build_problem(n)

    # Solving
    # Eigen's SimplicialCholesky is a direct solver for SPD matrices.
    # In SciPy, spsolve is the standard direct solver (uses SuperLU or CHOLMOD).
    # Alternatively, you can use spla.factorized(A) or an iterative solver like spla.cg.
    print("Solving the sparse linear system...")
    x = spla.spsolve(A, b)

    # Export the result to a file
    save_as_bitmap(x, n, sys.argv[1])

if __name__ == "__main__":
    main()