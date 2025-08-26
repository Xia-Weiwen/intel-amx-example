Example codes of matrix multiplication with Intel AMX

**File list:**
- int8-gemm-small.cpp: compute int8 matrix multiplication in small sizes
- int8-gemm-large.cpp: compute int8 matrix multiplication in large sizes
- bf16-gemm-small.cpp: compute bf16 matrix multiplication in small sizes
- bf16-gemm-large.cpp: compute bf16 matrix multiplication in large sizes

The examples with small shapes show how to manipulate with tile registers to compute a tiny GEMM.
The examples with large shapes show how to make full use of all tile registers and accumulate results of each small block of GEMM.

**How to build:**
- run `make`

**How to clean build:**
- run `make clean`
