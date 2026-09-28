#include <iostream>
#include <vector>

// Forward declarations
std::vector<double> csr_matvec(int m, const std::vector<int>& indptr, 
                               const std::vector<int>& indices, 
                               const std::vector<double>& data, 
                               const std::vector<double>& x);

std::vector<double> csc_matvec_transpose(int n, const std::vector<int>& indptr, 
                                         const std::vector<int>& indices, 
                                         const std::vector<double>& data, 
                                         const std::vector<double>& x);

int main() {
    // Example: 3x3 matrix A
    // [ 1  0  2 ]
    // [ 0  3  0 ]
    // [ 4  0  5 ]
    
    int m = 3; // rows
    int n = 3; // cols
    
    // CSR Arrays for A
    std::vector<int> indptr = {0, 2, 3, 5};
    std::vector<int> indices = {0, 2, 1, 0, 2};
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    std::vector<double> x = {1.0, 2.0, 3.0};
    
    // 1. Compute y = A * x
    std::vector<double> y = csr_matvec(m, indptr, indices, data, x);
    std::cout << "y = A * x: ";
    for (double val : y) std::cout << val << " ";
    std::cout << "\n";
    // Expected: 1*1 + 2*3 = 7, 3*2 = 6, 4*1 + 5*3 = 19 -> 7 6 19
    
    // 2. Compute y_T = A^T * x using the O(1) transpose trick
    std::vector<double> y_T = csc_matvec_transpose(n, indptr, indices, data, x);
    std::cout << "y_T = A^T * x: ";
    for (double val : y_T) std::cout << val << " ";
    std::cout << "\n";
    // Expected: 1*1 + 4*3 = 13, 3*2 = 6, 2*1 + 5*3 = 17 -> 13 6 17
    
    return 0;
}

// --- Function Implementations ---

std::vector<double> csr_matvec(int m, const std::vector<int>& indptr, 
                               const std::vector<int>& indices, 
                               const std::vector<double>& data, 
                               const std::vector<double>& x) {
    std::vector<double> y(m, 0.0);
    for (int i = 0; i < m; ++i) {
        double s = 0.0;
        for (int k = indptr[i]; k < indptr[i + 1]; ++k) {
            s += data[k] * x[indices[k]];
        }
        y[i] = s;
    }
    return y;
}

std::vector<double> csc_matvec_transpose(int n, const std::vector<int>& indptr, 
                                         const std::vector<int>& indices, 
                                         const std::vector<double>& data, 
                                         const std::vector<double>& x) {
    std::vector<double> y(n, 0.0);
    for (int j = 0; j < n; ++j) {
        for (int k = indptr[j]; k < indptr[j + 1]; ++k) {
            y[indices[k]] += data[k] * x[j];
        }
    }
    return y;
}