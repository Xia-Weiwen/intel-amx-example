/*
    Author: XIA, Weiwen <xia.weiwen@hotmail.com>

    This example shows how to compute matrix multiplication with Intel AMX instructions.
    The problem is C = A x B, where A's shape = [M, K], B's shape = [K, N], C's shape = [M, N]
    In this exmple, we use M = 12, K = 48, N = 12.
    With these small shapes, only three AMX tile registers are used and they are not fully filled.
*/

#include "common.h"

#define M 12
#define N 12
#define K 48

// Initialize tile config
// We need one tile for each of A, B and C
// A's shape = [12, 48] and dtype = int8. So, rows for A = 12 and colsb for A = 48.
// B's shape = [48, 12] and in VNNI layout [48/4, 12, 4].
// So, rows for B = 48 / 4 = 12, colsb for B = 12 * 4 = 48.
// C's shape = [12, 12] but in int32
// So, rows for C = 12 and colsb for C = 12 * sizeof(int32) = 48
void init_tile_config() {
    amx_tilecfg cfg_data;
    cfg_data.palette_id = 1;
    cfg_data.start_row = 0;
    // Configs for A, B and C. Note colsb comes before rows.
    // config for C
    cfg_data.colsb[0] = N * sizeof(int32_t);
    cfg_data.rows[0] = M;
    // config for A
    cfg_data.colsb[1] = K * sizeof(int8_t);
    cfg_data.rows[1] = M;
    // config for B
    cfg_data.colsb[2] = N * 4 * sizeof(int8_t);
    cfg_data.rows[2] = K / 4;

    _tile_loadconfig(&cfg_data);
}

void gemm_amx(int8_t* A, int8_t* B, int32_t* C) {
    // load A to tile 1
    _tile_loadd(1, A, /* stride */ K);
    // load B to tile 2
    _tile_loadd(2, B, /* stride */ N * 4);
    // clear tile 0 to hold results
    _tile_zero(0);
    // compute GEMM (dot product)
    _tile_dpbssd(0, 1, 2);
    // store results to C buffer
    _tile_stored(0, C, /* stride */ N * sizeof(int32_t));
}

int main() {

    std::cout << "=========================================\n";
    std::cout << "  Matrix multiplication with Intel AMX\n";
    std::cout << "=========================================\n";
    std::cout << "Shape: [" << M << ", " << K << "] x [" << K << ", " << N << "]\n";

    if (!init_amx()) return 1;

    int8_t A[M * K];
    int8_t B[K * N];
    int8_t B_VNNI[K * N];
    int32_t C[M * N];
    int32_t C_ref[M * N];

    std::cout << "init amx tile config...\n";
    init_tile_config();
    std::cout << "init buffer for A...\n";
    init_int8_buffer(A, M * K);
    std::cout << "init buffer for B...\n";
    init_int8_buffer(B, K * N);

    std::cout << "pack B to VNNI layout...\n";
    pack_B_to_vnni(B, N, K, B_VNNI);

    std::cout << "compute GEMM with ref impl...\n";
    int8_gemm_ref(A, B, C_ref, M, N, K);
    std::cout << "compute GEMM with AMX impl...\n";
    gemm_amx(A, B_VNNI, C);
    std::cout << "Check results...\n";
    check_results(C, C_ref, M, N);
    std::cout << "Release tiles...\n";
    _tile_release();
    std::cout << "Done\n";
    return 0;
}