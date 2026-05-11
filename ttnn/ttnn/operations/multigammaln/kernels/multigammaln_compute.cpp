// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for multigammaln at order p = 4.
//
// Per input tile:
//   1. Hold cb_input_tiles (cb_wait_front, NOT pop). The same input tile feeds
//      all four lgamma sub-phases.
//   2. For each offset in {0.0, 0.5, 1.0, 1.5}, run the fp32 lgamma recipe on
//      (x - offset) and pack the result into the matching intermediate CB
//      (cb_lgamma_a, _half, _one, _three_halves). This mirrors lgamma_kernel.cpp
//      line-for-line (Stirling + reflection-adjusted, suitable for fp32 inputs),
//      with a sub_unary_tile(offset) applied to every DEST slot that holds `x`
//      immediately after it is loaded. For offset == 0 the subtract is skipped
//      via `if constexpr` (subtracting +0 is a no-op).
//   3. Pop cb_input_tiles after all four lgammas have read it.
//   4. Sum the four lgamma terms into D0, add the compile-time constant
//      3 * log(pi) via add_unary_tile, and pack to cb_output_tiles.
//
// DEST budget: with fp32_dest_acc_en + half-sync, DEST holds 4 tiles per
// acquire/release. The per-offset lgamma recipe uses D0..D3 (matches the
// lgamma_kernel.cpp reference); the 4-way sum at the end also uses D0..D3.
//
// Domain note: multigammaln is well-defined for any a where every lgamma
// argument avoids non-positive integers. The standard domain is a > 1.5
// (where every lgamma argument is > 0). For out-of-domain a <= 1.5 the
// kernel does NOT branch on input value — values that hit a singularity
// fall through to NaN/inf naturally, matching `torch.special.multigammaln`.
//
// Precision note: every fp32 CB used by this kernel — cb_input_tiles,
// cb_output_tiles, and the four cb_lgamma_* intermediates — is configured
// with `UnpackToDestMode::UnpackToDestFp32` in the program descriptor. This
// is required: without it, `copy_tile(cb_lgamma_*, 0, slot)` in sub-phase B
// routes through SrcA/SrcB and may TF32-truncate. Empirically, that loss
// causes the four chained reflection-path lgammas to overflow to +inf for
// inputs that are within ~1e-3 of an integer boundary (e.g. a ≈ 0.5+ε),
// where torch returns a large but finite value. With UnpackToDestFp32 the
// fp32 mantissa survives the round-trip and the result matches torch.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/lgamma.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/rounding.h"
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/compute_kernel_api.h"

namespace {

constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_output_tiles = 16;
constexpr uint32_t cb_lgamma_a = 24;               // lgamma(x - 0.0)
constexpr uint32_t cb_lgamma_a_half = 25;          // lgamma(x - 0.5)
constexpr uint32_t cb_lgamma_a_one = 26;           // lgamma(x - 1.0)
constexpr uint32_t cb_lgamma_a_three_halves = 27;  // lgamma(x - 1.5)

constexpr float M_PI_F = 3.14159265358979323846f;

// IEEE-754 fp32 bit patterns for the per-offset subtractions.
constexpr uint32_t OFFSET_BITS_0_0 = 0x00000000u;  // 0.0f  — subtract elided
constexpr uint32_t OFFSET_BITS_0_5 = 0x3F000000u;  // 0.5f
constexpr uint32_t OFFSET_BITS_1_0 = 0x3F800000u;  // 1.0f
constexpr uint32_t OFFSET_BITS_1_5 = 0x3FC00000u;  // 1.5f

// Bit pattern of 3 * log(pi) ≈ 3.434189657547f (the (p*(p-1)/4)*log(pi)
// constant for p = 4).
constexpr uint32_t THREE_LOG_PI_BITS = 0x405BA32Eu;

// Per-offset lgamma recipe. Mirrors lgamma_kernel.cpp line-for-line, with a
// sub_unary_tile(offset_bits) applied to every DEST slot that holds `x`
// immediately after it is loaded. For offset_bits == 0 we skip the subtract.
//
// Pre: cb_input_tiles holds 1 tile at the front. Tile is NOT popped — caller
// pops after all four offsets have run.
template <uint32_t cb_out, uint32_t offset_bits>
ALWI void lgamma_with_offset() {
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();

    // ---- D0 = D1 = x_off ----
    copy_tile_to_dst_init_short(cb_input_tiles);
    copy_tile(cb_input_tiles, 0, 0);
    copy_tile(cb_input_tiles, 0, 1);

    if constexpr (offset_bits != 0u) {
        binop_with_scalar_tile_init();
        sub_unary_tile(0, offset_bits);
        sub_unary_tile(1, offset_bits);
    }

    // ---- D1 = z = (x_off < 0.5) ? 1 - x_off : x_off ----
    fill_tile_init();
    fill_tile(2, 0.5f);

    sub_binary_tile_init();
    sub_binary_tile(1, 2, 1);

    ltz_tile_init();
    ltz_tile(1);

    fill_tile_init();
    fill_tile(2, 1.0f);

    sub_binary_tile_init();
    sub_binary_tile(2, 0, 2);

    where_tile_init();
    where_tile<DataFormat::Float32>(1, 2, 0, 1);

    // ---- D1 = log(z) ----
    log_tile_init<false>();
    log_tile<false>(1);

    // ---- D0 = lgamma_stirling(x_off) ----
    lgamma_stirling_float_tile_init();
    lgamma_stirling_float_tile(0, 1, 0);

    // ---- D1 = sin(pi * frac(x_off)) ----
    fill_tile_init();
    fill_tile(2, M_PI_F);

    copy_tile_to_dst_init_short(cb_input_tiles);
    copy_tile(cb_input_tiles, 0, 1);
    if constexpr (offset_bits != 0u) {
        binop_with_scalar_tile_init();
        sub_unary_tile(1, offset_bits);
    }

    rounding_op_tile_init();
    frac_tile(1);

    mul_binary_tile_init();
    mul_binary_tile(1, 2, 1);

    sin_tile_init();
    sin_tile(1);

    // ---- D1 = log|sin(pi * frac(x_off))|, with integer-x_off → 0 mask ----
    copy_tile_to_dst_init_short(cb_input_tiles);
    copy_tile(cb_input_tiles, 0, 2);
    copy_tile(cb_input_tiles, 0, 3);
    if constexpr (offset_bits != 0u) {
        binop_with_scalar_tile_init();
        sub_unary_tile(2, offset_bits);
        sub_unary_tile(3, offset_bits);
    }

    rounding_op_tile_init();
    floor_tile(3);

    eq_binary_tile_init();
    eq_binary_tile(2, 3, 2);

    fill_tile_init();
    fill_tile(3, 0.0f);

    where_tile_init();
    where_tile<DataFormat::Float32>(2, 3, 1, 1);

    abs_tile_init();
    abs_tile(1);

    log_tile_init();
    log_tile(1);

    // ---- D2 reload = x_off (was clobbered by integer-mask) ----
    copy_tile_to_dst_init_short(cb_input_tiles);
    copy_tile(cb_input_tiles, 0, 2);
    if constexpr (offset_bits != 0u) {
        binop_with_scalar_tile_init();
        sub_unary_tile(2, offset_bits);
    }

    // ---- D0 = lgamma(x_off) with reflection for x_off < 0.5 ----
    lgamma_adjusted_tile_init();
    lgamma_adjusted_tile(0, 1, 2, 0);

    tile_regs_commit();
    tile_regs_wait();

    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);

    tile_regs_release();
}

// Sub-phase B: load all four lgamma terms into D0..D3, fold with three FPU
// add_binary_tile passes into D0, then add the compile-time constant
// 3 * log(pi) via add_unary_tile, and pack to cb_output_tiles.
//
// NOTE: this is implemented with raw APIs rather than the
// `sfpu_chain + sfpu_pipeline` helper because the chain framework default-
// constructs each op from its type list (see `sfpu_chain` at
// sfpu_helpers.hpp:1363–1371 → `ChainFromList<...>::type{}`). This discards
// scalar member values, so `AddScalar<Dst::D0>{scalar = K}` loses its `K`
// and adds 0 instead. Until the helper framework forwards op instances, the
// final +constant cannot be expressed as part of a chain. Documented as a
// refinement candidate in op_requirements.md.
ALWI void sum_and_add_const() {
    cb_wait_front(cb_lgamma_a, 1);
    cb_wait_front(cb_lgamma_a_half, 1);
    cb_wait_front(cb_lgamma_a_one, 1);
    cb_wait_front(cb_lgamma_a_three_halves, 1);

    cb_reserve_back(cb_output_tiles, 1);
    tile_regs_acquire();

    copy_tile_to_dst_init_short(cb_lgamma_a);
    copy_tile(cb_lgamma_a, 0, 0);
    copy_tile_to_dst_init_short(cb_lgamma_a_half);
    copy_tile(cb_lgamma_a_half, 0, 1);
    copy_tile_to_dst_init_short(cb_lgamma_a_one);
    copy_tile(cb_lgamma_a_one, 0, 2);
    copy_tile_to_dst_init_short(cb_lgamma_a_three_halves);
    copy_tile(cb_lgamma_a_three_halves, 0, 3);

    // D0 = D0 + D1 + D2 + D3
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    add_binary_tile(0, 2, 0);
    add_binary_tile(0, 3, 0);

    // D0 += 3 * log(pi)
    binop_with_scalar_tile_init();
    add_unary_tile(0, THREE_LOG_PI_BITS);

    tile_regs_commit();
    tile_regs_wait();

    pack_tile(0, cb_output_tiles);

    cb_pop_front(cb_lgamma_a, 1);
    cb_pop_front(cb_lgamma_a_half, 1);
    cb_pop_front(cb_lgamma_a_one, 1);
    cb_pop_front(cb_lgamma_a_three_halves, 1);
    cb_push_back(cb_output_tiles, 1);

    tile_regs_release();
}

}  // namespace

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    init_sfpu(cb_input_tiles, cb_output_tiles);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_input_tiles, 1);

        // Sub-phase A: four per-offset lgammas. Each call packs into its own
        // intermediate CB.
        lgamma_with_offset<cb_lgamma_a, OFFSET_BITS_0_0>();
        lgamma_with_offset<cb_lgamma_a_half, OFFSET_BITS_0_5>();
        lgamma_with_offset<cb_lgamma_a_one, OFFSET_BITS_1_0>();
        lgamma_with_offset<cb_lgamma_a_three_halves, OFFSET_BITS_1_5>();

        cb_pop_front(cb_input_tiles, 1);

        sum_and_add_const();
    }
}
