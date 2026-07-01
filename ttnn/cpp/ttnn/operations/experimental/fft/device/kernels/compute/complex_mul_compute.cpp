// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// complex_mul_compute.cpp — TRISC compute for the elementwise complex
// multiply used by the Bluestein chirp pre/post multiplies.
//
// For each tile pair (A, B) waited on the input CBs, this kernel computes
//     OUT_R = A_R * B_R - A_I * B_I
//     OUT_I = A_R * B_I + A_I * B_R
// using the SFPU binary-op path (full IEEE fp32, "precise" mode).
//
// One tile_regs_acquire/commit cycle per tile: all four input tiles are
// loaded into DST[0..3], products and sums stay in DST (no TMP CB traffic).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_api.h"

constexpr auto CB_A_R   = tt::CBIndex::c_0;
constexpr auto CB_A_I   = tt::CBIndex::c_1;
constexpr auto CB_B_R   = tt::CBIndex::c_2;
constexpr auto CB_B_I   = tt::CBIndex::c_3;
constexpr auto CB_OUT_R = tt::CBIndex::c_4;
constexpr auto CB_OUT_I = tt::CBIndex::c_5;

constexpr uint32_t DST_AR    = 0;
constexpr uint32_t DST_AI    = 1;
constexpr uint32_t DST_BR    = 2;
constexpr uint32_t DST_BI    = 3;
constexpr uint32_t DST_OUT_R = 0;
constexpr uint32_t DST_OUT_I = 2;

FORCE_INLINE void load_operand_tiles() {
    copy_tile_to_dst_init_short(CB_A_R);
    copy_tile(CB_A_R, 0, DST_AR);
    copy_tile_to_dst_init_short_with_dt(CB_A_R, CB_A_I);
    copy_tile(CB_A_I, 0, DST_AI);
    copy_tile_to_dst_init_short_with_dt(CB_A_I, CB_B_R);
    copy_tile(CB_B_R, 0, DST_BR);
    copy_tile_to_dst_init_short_with_dt(CB_B_R, CB_B_I);
    copy_tile(CB_B_I, 0, DST_BI);
}

// OUT_I = A_R*B_I + A_I*B_R, result in DST_OUT_I (DST_BR/DST_BI slots reused).
FORCE_INLINE void compute_out_imag() {
    mul_binary_tile_init();
    mul_binary_tile(DST_AI, DST_BR, DST_BI);   // AI*BR -> DST_BI
    copy_tile_to_dst_init_short_with_dt(CB_B_R, CB_B_I);
    copy_tile(CB_B_I, 0, DST_BR);            // reload B_I for A_R*B_I
    mul_binary_tile(DST_AR, DST_BR, DST_BR);  // A_R*B_I -> DST_BR
    add_binary_tile_init();
    add_binary_tile(DST_BR, DST_BI, DST_OUT_I);
}

// OUT_R = A_R*B_R - A_I*B_I, result in DST_OUT_R.
FORCE_INLINE void compute_out_real() {
    copy_tile_to_dst_init_short_with_dt(CB_A_I, CB_B_R);
    copy_tile(CB_B_R, 0, DST_BI);
    mul_binary_tile_init();
    mul_binary_tile(DST_AR, DST_BI, DST_OUT_R);  // A_R*B_R -> DST_AR
    copy_tile_to_dst_init_short_with_dt(CB_B_R, CB_B_I);
    copy_tile(CB_B_I, 0, DST_BI);
    mul_binary_tile(DST_AI, DST_BI, DST_AI);     // A_I*B_I -> DST_AI
    sub_binary_tile_init();
    sub_binary_tile(DST_OUT_R, DST_AI, DST_OUT_R);
}

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    unary_op_init_common(CB_A_R, CB_OUT_R);
    copy_tile_to_dst_init_short(CB_A_R);

    for (uint32_t k = 0; k < num_tiles; ++k) {
        cb_wait_front(CB_A_R, 1);
        cb_wait_front(CB_A_I, 1);
        cb_wait_front(CB_B_R, 1);
        cb_wait_front(CB_B_I, 1);

        tile_regs_acquire();
        load_operand_tiles();
        compute_out_imag();
        compute_out_real();
        tile_regs_commit();

        cb_reserve_back(CB_OUT_R, 1);
        cb_reserve_back(CB_OUT_I, 1);
        tile_regs_wait();
        pack_tile(DST_OUT_R, CB_OUT_R);
        pack_tile(DST_OUT_I, CB_OUT_I);
        tile_regs_release();
        cb_push_back(CB_OUT_R, 1);
        cb_push_back(CB_OUT_I, 1);

        cb_pop_front(CB_A_R, 1);
        cb_pop_front(CB_A_I, 1);
        cb_pop_front(CB_B_R, 1);
        cb_pop_front(CB_B_I, 1);
    }
}
