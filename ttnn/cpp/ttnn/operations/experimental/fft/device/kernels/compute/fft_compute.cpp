// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_compute.cpp — TRISC / compute
//

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_api.h"

constexpr auto CB_EVEN_R    = tt::CBIndex::c_0;
constexpr auto CB_EVEN_I    = tt::CBIndex::c_1;
constexpr auto CB_ODD_R     = tt::CBIndex::c_2;
constexpr auto CB_ODD_I     = tt::CBIndex::c_3;
constexpr auto CB_TW_R      = tt::CBIndex::c_4;
constexpr auto CB_TW_I      = tt::CBIndex::c_5;
constexpr auto CB_OUT0_R    = tt::CBIndex::c_6;
constexpr auto CB_OUT0_I    = tt::CBIndex::c_7;
constexpr auto CB_OUT1_R    = tt::CBIndex::c_8;
constexpr auto CB_OUT1_I    = tt::CBIndex::c_9;
constexpr auto CB_TMP_R     = tt::CBIndex::c_10;
constexpr auto CB_TMP_I     = tt::CBIndex::c_11;
constexpr auto CB_TW_ODD_R  = tt::CBIndex::c_12;
constexpr auto CB_TW_ODD_I  = tt::CBIndex::c_13;

constexpr uint32_t LOG2N = get_compile_time_arg_val(0);

enum : uint32_t { OP_ADD = 0, OP_SUB = 1, OP_MUL = 2 };

// SFPU binary op:  out[0] = a[0] <OP> b[0], pushed to `out`.
// Inputs are expected to be already waited-on; they are NOT popped here
// (caller controls lifetime). Full IEEE-fp32.
template <uint32_t OP>
FORCE_INLINE void sfpu_binop_push(uint32_t a, uint32_t b, uint32_t out) {
    tile_regs_acquire();

    copy_tile_to_dst_init_short(a);
    copy_tile(a, 0, 0);
    copy_tile_to_dst_init_short_with_dt(a, b);
    copy_tile(b, 0, 1);

    if      constexpr (OP == OP_ADD) { add_binary_tile_init(); add_binary_tile(0, 1, 0); }
    else if constexpr (OP == OP_SUB) { sub_binary_tile_init(); sub_binary_tile(0, 1, 0); }
    else if constexpr (OP == OP_MUL) { mul_binary_tile_init(); mul_binary_tile(0, 1, 0); }

    tile_regs_commit();

    cb_reserve_back(out, 1);
    tile_regs_wait();
    pack_tile(0, out);
    tile_regs_release();
    cb_push_back(out, 1);
}

FORCE_INLINE void cmul(
    uint32_t ar, uint32_t ai, uint32_t br, uint32_t bi,
    uint32_t outr, uint32_t outi)
{
    // outr = ar*br - ai*bi
    sfpu_binop_push<OP_MUL>(ar, br, CB_TMP_R);
    sfpu_binop_push<OP_MUL>(ai, bi, CB_TMP_I);
    cb_wait_front(CB_TMP_R, 1);
    cb_wait_front(CB_TMP_I, 1);
    sfpu_binop_push<OP_SUB>(CB_TMP_R, CB_TMP_I, outr);
    cb_pop_front(CB_TMP_R, 1);
    cb_pop_front(CB_TMP_I, 1);

    // outi = ar*bi + ai*br
    sfpu_binop_push<OP_MUL>(ar, bi, CB_TMP_R);
    sfpu_binop_push<OP_MUL>(ai, br, CB_TMP_I);
    cb_wait_front(CB_TMP_R, 1);
    cb_wait_front(CB_TMP_I, 1);
    sfpu_binop_push<OP_ADD>(CB_TMP_R, CB_TMP_I, outi);
    cb_pop_front(CB_TMP_R, 1);
    cb_pop_front(CB_TMP_I, 1);
}

void kernel_main() {
    // One-time init for copy_tile / SFPU pipeline.
    unary_op_init_common(CB_EVEN_R, CB_OUT0_R);
    copy_tile_to_dst_init_short(CB_EVEN_R);

    for (uint32_t s = 0; s < LOG2N; ++s) {
        cb_wait_front(CB_EVEN_R, 1);
        cb_wait_front(CB_EVEN_I, 1);
        cb_wait_front(CB_ODD_R,  1);
        cb_wait_front(CB_ODD_I,  1);
        cb_wait_front(CB_TW_R,   1);
        cb_wait_front(CB_TW_I,   1);

        // W * odd -> CB_TW_ODD
        cmul(CB_ODD_R, CB_ODD_I, CB_TW_R, CB_TW_I, CB_TW_ODD_R, CB_TW_ODD_I);

        cb_pop_front(CB_ODD_R, 1);
        cb_pop_front(CB_ODD_I, 1);
        cb_pop_front(CB_TW_R,  1);
        cb_pop_front(CB_TW_I,  1);

        cb_wait_front(CB_TW_ODD_R, 1);
        cb_wait_front(CB_TW_ODD_I, 1);

        // out0 = even + W*odd     out1 = even - W*odd
        sfpu_binop_push<OP_ADD>(CB_EVEN_R, CB_TW_ODD_R, CB_OUT0_R);
        sfpu_binop_push<OP_ADD>(CB_EVEN_I, CB_TW_ODD_I, CB_OUT0_I);
        sfpu_binop_push<OP_SUB>(CB_EVEN_R, CB_TW_ODD_R, CB_OUT1_R);
        sfpu_binop_push<OP_SUB>(CB_EVEN_I, CB_TW_ODD_I, CB_OUT1_I);

        cb_pop_front(CB_EVEN_R,   1);
        cb_pop_front(CB_EVEN_I,   1);
        cb_pop_front(CB_TW_ODD_R, 1);
        cb_pop_front(CB_TW_ODD_I, 1);
    }
}
