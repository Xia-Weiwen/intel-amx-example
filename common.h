#include <iostream>
#include <immintrin.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <cstdint>
#include <limits>
#include <random>
#include <cstring>

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

struct bfloat16 {
    uint16_t value;

    // Default constructor
    bfloat16() : value(0) {}

    // Construct from float
    bfloat16(float f) {
        uint32_t tmp;
        std::memcpy(&tmp, &f, sizeof(tmp));
        // Round to nearest even when truncating lower 16 bits
        uint32_t rounding_bias = ((tmp >> 16) & 1) + 0x7FFF;
        tmp += rounding_bias;
        value = static_cast<uint16_t>(tmp >> 16);
    }

    // Convert back to float
    operator float() const {
        uint32_t tmp = static_cast<uint32_t>(value) << 16;
        float f;
        std::memcpy(&f, &tmp, sizeof(f));
        return f;
    }
};

void init_bf16_buffer(bfloat16* buffer, int length) {
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> distr(-128, 127); // define the range
    for (int i = 0; i < length; ++i) {
        buffer[i] = bfloat16(distr(gen) / 120.0f);
    }
}

// assume B's shape = [K, N]
// Reorder from [K, N] to [K/vnni_size, N, vnni_size]
template <typename T>
void pack_B_to_vnni(T* in, int N, int K, T* out) {
    constexpr int vnni_size = std::is_same<T, int8_t>::value ? 4 : 2;
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            out[(k / vnni_size) * N * vnni_size + n * vnni_size + k % vnni_size] = in[k * N + n];
        }
    }
}

template <typename in_dtype, typename acc_dtype>
void gemm_ref(in_dtype* A, in_dtype* B, acc_dtype* C, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            acc_dtype acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += (acc_dtype)A[m * K + k] * (acc_dtype)B[k * N + n];
            }
            C[m * N + n] = acc;
        }
    }
}

template <typename T>
void check_results(T* C, T* C_ref, int M, int N, T tolerance = 0) {
    int error_count = 0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            if (std::fabs(C[m * N + n] - C_ref[m * N + n]) > tolerance) {
              std::cout << "error: ref=" << C[m * N + n] << " vs actual=" << C_ref[m * N + n]
                  << ", diff = " << std::fabs(C[m * N + n] - C_ref[m * N + n]) << std::endl;
              ++ error_count;
            }
        }
    }
    if (error_count == 0) std::cout << "OK\n";
    else std::cout << "\nFailed: " << error_count << "/" << (M * N) << " elements mismatch!\n";
}
