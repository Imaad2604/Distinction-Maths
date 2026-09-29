#!/usr/bin/env python3
"""Run both CSR benchmarks and compare performance"""
import subprocess
import time

def run_csr_bench():
    """Run the C++ CSR benchmark"""
    print("=" * 60)
    print("Running CSR Benchmark (C++)")
    print("=" * 60)
    try:
        result = subprocess.run(
            ["python3", "/home/imaad/Distinction-Maths/csr_bench.cpp"],
            capture_output=True, text=True, timeout=30
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running C++ benchmark: {e}")
        return False

def run_spmv_bench():
    """Run the SPMV benchmark (Python + C++)"""
    print("\n" + "=" * 60)
    print("Running SPMV Benchmark (Python + C++)")
    print("=" * 60)
    try:
        result = subprocess.run(
            ["python3", "/home/imaad/Distinction-Maths/csr_benchmark.py"],
            capture_output=True, text=True, timeout=60
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running SPMV benchmark: {e}")
        return False

if __name__ == "__main__":
    print("Starting benchmarks...")
    csr_ok = run_csr_bench()
    spmv_ok = run_spmv_bench()

    print("\n" + "=" * 60)
    if csr_ok and spmv_ok:
        print("Both benchmarks completed successfully!")
    else:
        print("Some benchmarks failed.")
