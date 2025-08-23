/*
    Author: XIA, Weiwen <xia.weiwen@hotmail.com>

    This example shows how to compute matrix multiplication with Intel AMX instructions.
    The problem is C = A x B, where A's shape = [M, K], B's shape = [K, N], C's shape = [M, N]
    In this exmple, we use M = 256, K = 256, N = 256.
    With these relatively large shapes, all AMX tile registers are used and fully filled.
    And we compute block by block with block_m & block_n = 32 and block_k = 64.
*/

#include "common.h"

#define M 256
#define N 256
#define K 256
#define BLOCK_M 32
#define BLOCK_N 32
#define BLOCK_K 64
#define MC (M / BLOCK_M)
#define NC (N / BLOCK_N)
#define KC (K / BLOCK_K)

// Initialize tile config to compute one block
// There totally 8 tile registers, TMM0 - TMM7.
// To make full use of AMX capability, We use 4 tiles for C and 2 tiles for A and B, respectively
// For each block, M = 32, N = 32, K = 64.
// A's shape = [32, 64] and dtype = int8. Two tiles for A. So, rows for A = 32 / 2 = 16 and colsb for A = 64.
// B's shape = [64, 32] and in VNNI layout [64/4, 32, 4] = [16, 128]. Two tiles for B.
// So, rows for B = 16, colsb for B = 128 / 2 = 64.
// C's shape = [32, 32] but in int32 and we need 4 tiles to hold them.
// So, rows for C = 32 / 2 = 16 and colsb for C = 32 / 2 * sizeof(int32) = 64.
void init_tile_config() {
    amx_tilecfg cfg_data;
    cfg_data.palette_id = 1;
    cfg_data.start_row = 0;
    // Configs for A, B and C. Note colsb comes before rows.
    // rows and colsb for A, B and C are all the same.
    // But we separate them to make it clearer.
    // config for C
    int i = 0;
    for (; i < 4; ++i) {
        cfg_data.colsb[i] = 64;
        cfg_data.rows[i] = 16;
    }
    // config for A
    for (; i < 6; ++i) {
        cfg_data.colsb[i] = 64;
        cfg_data.rows[i] = 16;
    }
    // config for B
    for (; i < 8; ++i) {
        cfg_data.colsb[i] = 64;
        cfg_data.rows[i] = 16;
    }

    _tile_loadconfig(&cfg_data);
}

// Pack B to blocked layout in memory and in each block, data are in VNNI layout
// [K, N] -> [K/block_k, block_k, N/block_n, block_n] -> [K/block_k, N/block_n, block_k, block_n]
// And we pack each block of [block_k, block_n] to VNNI layout.
// We can definitely fuse the two steps (i.e., blocking & VNNI-packing) to avoid the intermediate buffer.
// But here we don't, to make it simple.
#define IN_IDX(kc, nc, kb, nb) ((kc * BLOCK_K + kb) * N + (nc * BLOCK_N + nb))
void pack_B(int8_t* in, int8_t* out) {
    for (int kc = 0; kc < KC; ++kc) {
        for (int nc = 0; nc < NC; ++nc) {
            int8_t block_B_buffer[BLOCK_K * BLOCK_N];
            for (int kb = 0; kb < BLOCK_K; ++kb) {
                for (int nb = 0; nb < BLOCK_N; ++nb) {
                    block_B_buffer[kb * BLOCK_N + nb] = in[IN_IDX(kc, nc, kb, nb)];
                }
            }
            pack_B_to_vnni(block_B_buffer, BLOCK_N, BLOCK_K, &out[kc * BLOCK_K * N + nc * BLOCK_K * BLOCK_N]);
        }
    }
}

void gemm_amx(int8_t* A, int8_t* B, int32_t* C) {
    // Compute with a nested loop
    for (int mc = 0; mc < MC; ++mc) {
        for (int nc = 0; nc < NC; ++nc) {
            // compute block by block and accumulate along K
            // 1. clear C tiles
            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);
            // 2. loop over K
            for (int kc = 0; kc < KC; ++kc) {
                // 2.1 load a block of A [32, 64] to tile 4 & 5 (different M)
                _tile_loadd(4, A + mc * BLOCK_M * K + kc * BLOCK_K, /* stride */ K);
                _tile_loadd(5, A + (mc * BLOCK_M + 16) * K + kc * BLOCK_K, /* stride */ K);
                // 2.2 load a block of B [BLOCK_K/4, BLOCK_N, 4] -> [16, 128] to tile 6 & 7 (different N)
                //     B's shape = [K/block_k, N/block_n, block_k/4, block_n, 4]
                _tile_loadd(6, B + kc * BLOCK_K * N + nc * BLOCK_N * BLOCK_K, /* stride */ BLOCK_N * 4);
                _tile_loadd(7, B + kc * BLOCK_K * N + nc * BLOCK_N * BLOCK_K + 64, /* stride */ BLOCK_N * 4);
                // 2.3 compute GEMM of one block (dot product)
                //         N
                //   +-----+-----+
                //   |  0  |  1  |
                // M +-----+-----+
                //   |  2  |  3  |
                //   +-----+-----+
                _tile_dpbssd(0, 4, 6);
                _tile_dpbssd(1, 4, 7);
                _tile_dpbssd(2, 5, 6);
                _tile_dpbssd(3, 5, 7);
            }
            // 3. store results to C buffer
            _tile_stored(0, C + mc * BLOCK_M * N + nc * BLOCK_N, /* stride */ N * sizeof(int32_t));
            _tile_stored(1, C + mc * BLOCK_M * N + nc * BLOCK_N + 16, /* stride */ N * sizeof(int32_t));
            _tile_stored(2, C + (mc * BLOCK_M + 16) * N + nc * BLOCK_N, /* stride */ N * sizeof(int32_t));
            _tile_stored(3, C + (mc * BLOCK_M + 16) * N + nc * BLOCK_N + 16, /* stride */ N * sizeof(int32_t));
        }
    }
}

int main() {

    std::cout << "=========================================\n";
    std::cout << "  Matrix multiplication with Intel AMX\n";
    std::cout << "=========================================\n";
    std::cout << "Shape: [" << M << ", " << K << "] x [" << K << ", " << N << "]\n";

    if (!init_amx()) return 1;

    int8_t A[M * K];
    int8_t B[K * N];
    int8_t B_packed[K * N];
    int32_t C[M * N];
    int32_t C_ref[M * N];

    std::cout << "init amx tile config...\n";
    init_tile_config();
    std::cout << "init buffer for A...\n";
    init_int8_buffer(A, M * K);
    std::cout << "init buffer for B...\n";
    init_int8_buffer(B, K * N);

    std::cout << "pack B to blocked & VNNI layout...\n";
    pack_B(B, B_packed);

    std::cout << "compute GEMM with ref impl...\n";
    int8_gemm_ref(A, B, C_ref, M, N, K);
    std::cout << "compute GEMM with AMX impl...\n";
    gemm_amx(A, B_packed, C);
    std::cout << "Check results...\n";
    check_results(C, C_ref, M, N);
    std::cout << "Release tiles...\n";
    _tile_release();
    std::cout << "Done\n";
    return 0;
}