#include <cstdio>
#include <chrono>
#include <vector>
#include <cstdint>

int main() {
    long long n_rows_ll = 0, n_cols_ll = 0, nnz_ll = 0;
    if (std::scanf("%lld %lld %lld", &n_rows_ll, &n_cols_ll, &nnz_ll) != 3) {
        std::fprintf(stderr,
            "csr_bench: expected input on stdin.\n"
            "  Format: n_rows n_cols nnz\\n"
            "          indptr[0..n_rows]\\n"
            "          indices[0..nnz-1]\\n"
            "          data[0..nnz-1]\\n"
            "          x[0..n_cols-1]\\n"
            "  This binary is normally invoked by spmv_languages.py.\n");
        return 1;
    }

    const int64_t n_rows = n_rows_ll;
    const int64_t n_cols = n_cols_ll;
    const int64_t nnz    = nnz_ll;

    if (n_rows <= 0 || n_cols <= 0 || nnz < 0) {
        std::fprintf(stderr, "csr_bench: invalid dimensions.\n");
        return 1;
    }

    std::vector<int64_t> indptr(n_rows + 1);
    std::vector<int32_t> indices(nnz);
    std::vector<double>  data(nnz);
    std::vector<double>  x(n_cols);
    std::vector<double>  y(n_rows, 0.0);

    for (int64_t i = 0; i <= n_rows; ++i) {
        long long v = 0;
        if (std::scanf("%lld", &v) != 1) {
            std::fprintf(stderr, "csr_bench: failed to read indptr[%lld]\n",
                         (long long)i);
            return 1;
        }
        indptr[i] = v;
    }
    for (int64_t k = 0; k < nnz; ++k) {
        int v = 0;
        if (std::scanf("%d", &v) != 1) {
            std::fprintf(stderr, "csr_bench: failed to read indices[%lld]\n",
                         (long long)k);
            return 1;
        }
        indices[k] = v;
    }
    for (int64_t k = 0; k < nnz; ++k) {
        if (std::scanf("%lf", &data[k]) != 1) {
            std::fprintf(stderr, "csr_bench: failed to read data[%lld]\n",
                         (long long)k);
            return 1;
        }
    }
    for (int64_t j = 0; j < n_cols; ++j) {
        if (std::scanf("%lf", &x[j]) != 1) {
            std::fprintf(stderr, "csr_bench: failed to read x[%lld]\n",
                         (long long)j);
            return 1;
        }
    }

    for (int64_t i = 0; i < n_rows; ++i) {
        double s = 0.0;
        for (int64_t k = indptr[i]; k < indptr[i + 1]; ++k) {
            s += data[k] * x[indices[k]];
        }
        y[i] = s;
    }

    const int reps = 20;
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
        for (int64_t i = 0; i < n_rows; ++i) {
            double s = 0.0;
            for (int64_t k = indptr[i]; k < indptr[i + 1]; ++k) {
                s += data[k] * x[indices[k]];
            }
            y[i] = s;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;

    double checksum = 0.0;
    for (int64_t i = 0; i < n_rows; ++i) checksum += y[i];


    std::printf("CSR_CPP_TIME_MS %.6f\n", ms);
    std::printf("CSR_CPP_CHECKSUM %.10f\n", checksum);
    return 0;
}