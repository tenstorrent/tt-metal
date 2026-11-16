// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// moe_decode gate|up compute (TRISC): g = x . Wg[:, j], u = x . Wu[:, j] (custom_mm, 1x32 activation tiles, LoFi,
// fp32 dest), h = gelu(g) * u * w_s (w_s: the slot's routing weight, from BRISC through the mailbox), one 1x32 tile.
// The weights arrive as one [gate ks | up ks] chunk per DRAM bank in the reader's rotated bank order; each chunk is
// accumulated as it lands (split accumulation, merged on the last chunk).
// compile-time args: [cb_x, cb_w, cb_out, kt, banks]
// runtime args: [first_bank]
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"

// A 1x32 matmul result occupies the first rows of DST faces 0 and 1: SFPU over rows only (VectorMode::R).
constexpr uint32_t kRowIters = 2;

ALWI void gelu_rows(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_gelu, (false, DST_ACCUM_MODE, kRowIters), idst, VectorMode::R));
}

ALWI void mul_rows(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, kRowIters, DST_ACCUM_MODE),
        a,
        b,
        out,
        VectorMode::R)));
}

ALWI void scale_rows(uint32_t idst, uint32_t s) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, MUL_UNARY, kRowIters, DST_ACCUM_MODE),
        idst,
        VectorMode::R,
        s));
}

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);
    constexpr uint32_t banks = get_compile_time_arg_val(4);
    constexpr uint32_t ks = kt / banks;
    const uint32_t first_bank = get_arg_val<uint32_t>(0);
    constexpr bool transpose = false;
    constexpr auto dest_layout = CustomMmDestLayout::DefaultSplitAcc;
    custom_mm_block_init<transpose, dest_layout>(cb_x, cb_w, cb_out);
    uint32_t score = 0;
    MATH(score = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    cb_wait_front(cb_x, kt);
    tile_regs_acquire();
    for (uint32_t i = 0; i < banks; ++i) {
        const uint32_t b = (first_bank + i) % banks;
        cb_wait_front(cb_w, 2 * ks);
        if (i + 1 < banks) {
            custom_mm_block<dest_layout, false>(cb_x, cb_w, b * ks, 0, 0, ks);
            custom_mm_block<dest_layout, false>(cb_x, cb_w, b * ks, ks, 1, ks);
        } else {
            custom_mm_block<dest_layout, true>(cb_x, cb_w, b * ks, 0, 0, ks);
            custom_mm_block<dest_layout, true>(cb_x, cb_w, b * ks, ks, 1, ks);
        }
        cb_pop_front(cb_w, 2 * ks);
    }
    gelu_tile_init<false>();
    gelu_rows(0);
    mul_binary_tile_init();
    mul_rows(0, 1, 0);
    binop_with_scalar_tile_init();
    scale_rows(0, score);
    tile_regs_commit();
    cb_reserve_back(cb_out, 1);
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
    cb_pop_front(cb_x, kt);
    custom_mm_block_uninit<dest_layout>(cb_out);
}
