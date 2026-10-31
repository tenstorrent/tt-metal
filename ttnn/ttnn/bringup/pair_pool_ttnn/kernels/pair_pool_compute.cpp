// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.bringup.pair_pool compute. Per unit: cb_in's [kv0 | g0 | kv1 | g1] (UnpackToDestFp32, exact) -> DEST 0..3, then
// lane-wise fp32 SFPU, the op chain's steps (deepseek_v41_flash_d_p tt/compressor.py TtCompressor._pool) with the
// stock ops' SFPU bodies, each step its own fp32 rounding (results go through DEST between steps):
//     m = max(g0, g1); e_k = exp(g_k - m) (exp: the stock fp32 accurate body); den = e0 + e1;
//     w_k = e_k / den (the stock fp32 SFPU divide body); out = kv0 w0 + kv1 w1 (two products, then their sum)
// -> DEST 0 -> cb_out. DEST 4 holds den.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"

#ifdef TRISC_MATH
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#endif

constexpr uint32_t cb_in = get_compile_time_arg_val(0);
constexpr uint32_t cb_out = get_compile_time_arg_val(1);

#ifdef TRISC_MATH
namespace pair_pool_sfpu {

using namespace sfpi;
constexpr int T = 32;  // SFPU vectors a tile

// ckernel_sfpu_binary.h calculate_sfpu_binary_div, its fp32 DEST branch (one vector)
sfpi_inline vFloat div_fp32(const vFloat in0, const vFloat in1) {
    vFloat ma = setexp(in0, 127);
    vFloat mb = setexp(in1, 127);
    vFloat r = ckernel::sfpu::sfpu_reciprocal_iter<1, true>(mb);
    vFloat q = ma * r;
    vFloat residual = ma - q * mb;
    q = q + residual * r;
    vInt ea = exexp(in0, ExponentMode::Biased);
    vInt eb = exexp(in1);
    vInt exponent = ea - eb + exexp(q, ExponentMode::Biased);
    vInt half = as<vInt>(as<vUInt>(exponent) >> 1);
    vFloat result = setexp(q, half) * setexp(vFloat(1.0f), exponent - half);
    v_if(!(ea >= 1 && ea < 255 && eb >= -126 && eb < 128)) {
        vFloat scale = setman(as<vFloat>(~as<vInt>(in1)), 0);
        vFloat nan_divisor = as<vFloat>(mask(setsgn(in1, 0) > std::numeric_limits<float>::infinity()));
        result = nan_divisor - in0 * scale;
    }
    v_endif;
    return result;
}

void pool() {
    // e_k = exp(g_k - max(g0, g1)) -> DEST 1 (e0), 3 (e1)
#pragma GCC unroll 1
    for (int v = 0; v < T; ++v) {
        const vFloat g0 = dst_reg[T + v];
        const vFloat g1 = dst_reg[3 * T + v];
        vFloat m = g0;
        v_if(g0 < g1) { m = g1; }
        v_endif;
        dst_reg[T + v] = g0 - m;
        dst_reg[3 * T + v] = g1 - m;
    }
#pragma GCC unroll 1
    for (int v = 0; v < T; ++v) {
        dst_reg[T + v] = ckernel::sfpu::_sfpu_exp_accurate_<true>(dst_reg[T + v]);
        dst_reg[3 * T + v] = ckernel::sfpu::_sfpu_exp_accurate_<true>(dst_reg[3 * T + v]);
    }
    ckernel::sfpu::sfpu_reciprocal_init<false>();  // (the divide's Newton step constant)
#pragma GCC unroll 1
    for (int v = 0; v < T; ++v) {
        dst_reg[4 * T + v] = vFloat(dst_reg[T + v]) + vFloat(dst_reg[3 * T + v]);      // den
        dst_reg[T + v] = div_fp32(dst_reg[T + v], dst_reg[4 * T + v]);                 // w0
        dst_reg[3 * T + v] = div_fp32(dst_reg[3 * T + v], dst_reg[4 * T + v]);         // w1
        dst_reg[v] = vFloat(dst_reg[v]) * vFloat(dst_reg[T + v]);                      // kv0 w0
        dst_reg[2 * T + v] = vFloat(dst_reg[2 * T + v]) * vFloat(dst_reg[3 * T + v]);  // kv1 w1
        dst_reg[v] = vFloat(dst_reg[v]) + vFloat(dst_reg[2 * T + v]);
    }
}

}  // namespace pair_pool_sfpu
#endif

using namespace ckernel;

void kernel_main() {
    uint32_t first, count;
    {
        const uint32_t per = get_common_arg_val<uint32_t>(0), extra = get_common_arg_val<uint32_t>(1);
        const uint32_t c = static_cast<uint32_t>(get_absolute_logical_y()) * PAIR_POOL_GX + get_absolute_logical_x();
        count = per + (c < extra ? 1u : 0u);
        first = c * per + (c < extra ? c : extra);
    }
    (void)first;
    compute_kernel_hw_startup(cb_in, cb_out);
    MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>()));
    copy_init(cb_in);
    pack_init(cb_out);
    for (uint32_t u = 0; u < count; ++u) {
        cb_wait_front(cb_in, 4);
        tile_regs_acquire();
        for (uint32_t t = 0; t < 4; ++t) {
            copy_tile(cb_in, t, t);
        }
        MATH((_llk_math_eltwise_unary_sfpu_params_(pair_pool_sfpu::pool, 0, VectorMode::None)));
        tile_regs_commit();
        cb_pop_front(cb_in, 4);
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
}
