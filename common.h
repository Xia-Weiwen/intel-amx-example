#include <iostream>
#include <immintrin.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <cstdint>
#include <random>

#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18
#define XFEATURE_MASK_XTILECFG (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)
#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023

bool init_amx() {
  unsigned long bitmask = 0;
  // Request permission to use AMX instructions
  long rc = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
  if (rc) {
      std::cout << "Failed to enable AMX\n";
      return false;
  }
  // Check if the system supports AMX instructions
  rc = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
  if (rc) {
      std::cout << "AMX is not supported on your hardware\n";
      return false;
  }
  if (bitmask & XFEATURE_MASK_XTILE) {
      std::cout << "AMX is supported on your hardware and it's enabled\n";
      return true;
  }
  return false;
}

// Define tile config data structure
struct amx_tilecfg {
    uint8_t palette_id = 0;
    uint8_t start_row = 0;
    uint8_t reserved_0[14] = {0};
    uint16_t colsb[16] = {0};
    uint8_t rows[16] = {0};
};

void init_int8_buffer(int8_t* buffer, int length) {
    // https://stackoverflow.com/questions/7560114/random-number-c-in-some-range
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> distr(-128, 127); // define the range
    for (int i = 0; i < length; ++i) {
        buffer[i] = distr(gen);
    }
}

// assume B's shape = [K, N]
// Reorder from [K, N] to [K/4, N, 4]
void pack_B_to_vnni(int8_t* in, int N, int K, int8_t* out) {
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            out[(k / 4) * N * 4 + n * 4 + k % 4] = in[k * N + n];
        }
    }
}

void int8_gemm_ref(int8_t* A, int8_t* B, int32_t* C, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = acc;
        }
    }
}

void check_results(int32_t* C, int32_t* C_ref, int M, int N) {
    int error_count = 0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            if (C[m * N + n] != C_ref[m * N + n]) ++ error_count;
        }
    }
    if (error_count == 0) std::cout << "OK\n";
    else std::cout << "\nFailed: " << error_count << "/" << (M * N) << " elements mismatch!\n";
}
