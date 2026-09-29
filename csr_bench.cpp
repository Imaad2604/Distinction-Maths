#include <cstdio>
#include <vector>
#include <chrono>

int main() {
    int n_rows, n_cols;
    long long nnz;
    if (scanf("%d %d %lld", &n_rows, &n_cols, &nnz) != 3) return 1;

    std::vector<int> indptr(n_rows + 1);
    std::vector<int> indices(nnz);
    std::vector<double> data(nnz);
    std::vector<double> x(n_cols);

    for (int i = 0; i <= n_rows; ++i) scanf("%d", &indptr[i]);
    for (long long k = 0; k < nnz; ++k) scanf("%d", &indices[k]);
    for (long long k = 0; k < nnz; ++k) scanf("%lf", &data[k]);
    for (int j = 0; j < n_cols; ++j) scanf("%lf", &x[j]);

    // Warmup
    std::vector<double> y(n_rows, 0.0);
    for (int i = 0; i < n_rows; ++i) {
        double s = 0.0;
        for (int k = indptr[i]; k < indptr[i + 1]; ++k)
            s += data[k] * x[indices[k]];
        y[i] = s;
    }

    // Timed run (50 reps)
    int reps = 50;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < reps; ++r) {
        for (int i = 0; i < n_rows; ++i) {
            double s = 0.0;
            for (int k = indptr[i]; k < indptr[i + 1]; ++k)
                s += data[k] * x[indices[k]];
            y[i] = s;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;

    double checksum = 0.0;
    for (double v : y) checksum += v;

    printf("CSR_CPP_TIME_MS %.6f\n", ms);
    printf("CSR_CPP_CHECKSUM %.17g\n", checksum);
    return 0;
}
