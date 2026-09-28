def coo_to_csr(m, rows, cols, vals):
    """
    Converts COO format (rows, cols, vals) to CSR format (indptr, indices, data).
    
    Parameters:
    m (int): Number of rows in the matrix.
    rows (list of int): Row indices of non-zero elements.
    cols (list of int): Column indices of non-zero elements.
    vals (list of float): Values of non-zero elements.
    
    Returns:
    tuple: (indptr, indices, data)
    """

    indptr = [0] * (m + 1)
    for i in rows:
        indptr[i + 1] += 1

    for i in range(m):
        indptr[i + 1] += indptr[i]

    nxt = indptr[:-1].copy() 
    
    nnz = len(vals)
    indices = [0] * nnz
    data = [0.0] * nnz

    for i, j, v in zip(rows, cols, vals):
        p = nxt[i]                
        indices[p] = j            
        data[p] = v                
        nxt[i] += 1                

    return indptr, indices, data

if __name__ == "__main__":
    
    m = 4
    rows = [0, 0, 1, 1, 2, 2, 2, 3]
    cols = [0, 3, 0, 1, 1, 2, 3, 3]
    vals = [10, 2, 3, 9, 7, 8, 7, 5]

    print("--- Input (COO) ---")
    print("Rows:   ", rows)
    print("Cols:   ", cols)
    print("Vals:   ", vals)
    print()

    # Run the conversion
    indptr, indices, data = coo_to_csr(m, rows, cols, vals)

    print("--- Output (CSR) ---")
    print("Indptr: ", indptr)
    print("Indices:", indices)
    print("Data:   ", data)