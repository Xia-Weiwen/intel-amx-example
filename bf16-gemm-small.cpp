/*
    Author: XIA, Weiwen <xia.weiwen@hotmail.com>

    This example shows how to compute matrix multiplication with Intel AMX instructions.
    The problem is C = A x B, where A's shape = [M, K], B's shape = [K, N], C's shape = [M, N]
    In this exmple, A and B are in bf16 and C in float. And we use M = 12, K = 24, N = 12.
    With these small shapes, only three AMX tile registers are used and they are not fully filled.
*/

#include "common.h"

#define M 12
#define N 12
#define K 24

// Initialize tile config
// We need one tile for each of A, B and C
// A's shape = [12, 24] and dtype = bf16.
// So, rows for A = 12 and colsb for A = 24 * sizeof(bf16) = 24 * 2 = 48.
// B's shape = [24, 12] and in VNNI layout [24/2, 12, 2].
// So, rows for B = 24 / 2 = 12, colsb for B = 12 * 2 * sizeof(bf16) = 48.
// C's shape = [12, 12] but in float
// So, rows for C = 12 and colsb for C = 12 * sizeof(float) = 48
void init_tile_config() {
    amx_tilecfg cfg_data;
    cfg_data.palette_id = 1;
    cfg_data.start_row = 0;
    // Configs for A, B and C. Note colsb comes before rows.
    // config for C
    cfg_data.colsb[0] = N * sizeof(float);
    cfg_data.rows[0] = M;
    // config for A
    cfg_data.colsb[1] = K * sizeof(bfloat16);
    cfg_data.rows[1] = M;
    // config for B
    cfg_data.colsb[2] = N * 2 * sizeof(bfloat16);
    cfg_data.rows[2] = K / 2;

    _tile_loadconfig(&cfg_data);
}

void gemm_amx(bfloat16* A, bfloat16* B, float* C) {
    // load A to tile 1
    _tile_loadd(1, A, /* stride */ K * sizeof(bfloat16));
    // load B to tile 2
    _tile_loadd(2, B, /* stride */ N * 2 * sizeof(bfloat16));
    // clear tile 0 to hold results
    _tile_zero(0);
    // compute GEMM (dot product)
    _tile_dpbf16ps(0, 1, 2);
    // store results to C buffer
    _tile_stored(0, C, /* stride */ N * sizeof(float));
}

int main() {

    std::cout << "=========================================\n";
    std::cout << "  Matrix multiplication with Intel AMX\n";
    std::cout << "=========================================\n";
    std::cout << "Data type: bf16 * bf16 -> float\n";
    std::cout << "Shape: [" << M << ", " << K << "] x [" << K << ", " << N << "]\n";

    if (!init_amx()) return 1;

    bfloat16 A[M * K];
    bfloat16 B[K * N];
    bfloat16 B_VNNI[K * N];
    float C[M * N];
    float C_ref[M * N];

    std::cout << "init amx tile config...\n";
    init_tile_config();
    std::cout << "init buffer for A...\n";
    init_bf16_buffer(A, M * K);
    std::cout << "init buffer for B...\n";
    init_bf16_buffer(B, K * N);

    std::cout << "pack B to VNNI layout...\n";
    pack_B_to_vnni(B, N, K, B_VNNI);

    std::cout << "compute GEMM with ref impl...\n";
    gemm_ref(A, B, C_ref, M, N, K);
    std::cout << "compute GEMM with AMX impl...\n";
    gemm_amx(A, B_VNNI, C);
    std::cout << "Check results...\n";
    check_results(C, C_ref, M, N, 1e-5f);
    std::cout << "Release tiles...\n";
    _tile_release();
    std::cout << "Done\n";
    return 0;
}