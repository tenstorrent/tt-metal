// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_silu.h"
#include "llk_assert.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

/**
 * @brief Fused per-channel causal conv1d (4-tap weighted sum) + SiLU.
 *
 * @tparam APPROXIMATION_MODE Selects the approximate math path for the SiLU half (unused by
 *   the FMA half, kept for signature symmetry with other SFPU ops).
 * @tparam is_fp32_dest_acc_en When false, both outputs are rounded to bf16 on store (matches
 *   the `calculate_addcmul()`/`calculate_lerp()` convention); when true, both outputs are
 *   stored at full fp32 precision.
 * @tparam ITERATIONS Number of 32-lane rows to process *within one face* (one row per loop
 *   iteration). A full 32x32 tile is 4 faces of 16x16; the caller must invoke this function once
 *   per face (e.g. via `_llk_math_eltwise_sfpu_apply_vector_mode_(..., VectorMode::RC)`, or an
 *   explicit 4x loop with `_llk_math_eltwise_sfpu_inc_dst_face_addr_()` between calls) to cover
 *   the whole tile -- a single call only ever advances within the current face's window.
 * @param dst_index_wa Tile holding per-channel weight `a` (multiplies `x`).
 * @param dst_index_wb Tile holding per-channel weight `b` (multiplies `y`).
 * @param dst_index_wc Tile holding per-channel weight `c` (multiplies `z`).
 * @param dst_index_wd Tile holding per-channel weight `d` (multiplies `w`).
 * @param dst_index_x Oldest-but-one cache entry.
 * @param dst_index_y Middle cache entry.
 * @param dst_index_z Newest cache entry (about to be evicted from the 3-wide cache).
 * @param dst_index_w Newest sample, already produced by a preceding matmul.
 * @param dst_index_cache_out Output tile for `new_cache = a*x + b*y + c*z + d*w`. May safely
 *   equal any of `dst_index_wa/wb/wc/wd` (all 4 weight tiles are fully consumed into local
 *   registers before this function's first store), matching the `dst_index_out ==
 *   dst_index_in0` reuse convention `calculate_addcmul()`/`calculate_where()` already rely on.
 * @param dst_index_silu_out Output tile for `SiLU(new_cache)`; same reuse rule as
 *   `dst_index_cache_out`, and must differ from it.
 *
 * @note Implements the KDA (Kimi Delta Attention) conv1d node from
 *   tt-blaze#2388/tt-blaze#2429: `new_cache` is the updated causal-conv accumulator (per
 *   channel: `a*x + b*y + c*z + d*w`, all four operands are per-channel BF16 tensors, not
 *   scalars); one output is `SiLU(new_cache)`, the other is the 3-wide cache shift
 *   `[new_cache, x, y]`. This function computes only the two values that require new math
 *   (`new_cache`, `SiLU(new_cache)`); the caller re-packs `x`/`y` unchanged to assemble the
 *   3-wide shifted cache, since that half of the update is a pure data-movement concern, not
 *   an SFPU compute one. All operands are expected already resident in Dest (unpacked/copied by
 *   the caller), following the same `sfpi::dst_reg[dst_index*32 + row]` explicit-addressing
 *   idiom as `calculate_addcmul()`/`calculate_lerp()`; unlike `addcmul`'s scalar `value`, every
 *   weight here is a full per-channel tensor, so all four taps are genuine tensor*tensor FMAs,
 *   accumulated once in registers (a single load of each of the 8 inputs, no intermediate Dest
 *   round-trips) before the two final stores. @ref _sigmoid_piecewise_linear_positive_ is
 *   reused unmodified for the SiLU half.
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void _calculate_causal_conv1d_silu_(
    const std::uint32_t dst_index_wa,
    const std::uint32_t dst_index_wb,
    const std::uint32_t dst_index_wc,
    const std::uint32_t dst_index_wd,
    const std::uint32_t dst_index_x,
    const std::uint32_t dst_index_y,
    const std::uint32_t dst_index_z,
    const std::uint32_t dst_index_w,
    const std::uint32_t dst_index_cache_out,
    const std::uint32_t dst_index_silu_out)
{
    LLK_ASSERT(dst_index_cache_out != dst_index_silu_out, "dst_index_cache_out and dst_index_silu_out must be distinct Dest tiles");

    constexpr std::uint32_t dst_tile_size_sfpi = 32;

    const std::uint32_t off_wa = dst_index_wa * dst_tile_size_sfpi;
    const std::uint32_t off_wb = dst_index_wb * dst_tile_size_sfpi;
    const std::uint32_t off_wc = dst_index_wc * dst_tile_size_sfpi;
    const std::uint32_t off_wd = dst_index_wd * dst_tile_size_sfpi;
    const std::uint32_t off_x  = dst_index_x * dst_tile_size_sfpi;
    const std::uint32_t off_y  = dst_index_y * dst_tile_size_sfpi;
    const std::uint32_t off_z  = dst_index_z * dst_tile_size_sfpi;
    const std::uint32_t off_w  = dst_index_w * dst_tile_size_sfpi;

    const std::uint32_t off_cache_out = dst_index_cache_out * dst_tile_size_sfpi;
    const std::uint32_t off_silu_out  = dst_index_silu_out * dst_tile_size_sfpi;

    for (int d = 0; d < ITERATIONS; d++)
    {
        const sfpi::vFloat wa = sfpi::dst_reg[off_wa];
        const sfpi::vFloat wb = sfpi::dst_reg[off_wb];
        const sfpi::vFloat wc = sfpi::dst_reg[off_wc];
        const sfpi::vFloat wd = sfpi::dst_reg[off_wd];
        const sfpi::vFloat x  = sfpi::dst_reg[off_x];
        const sfpi::vFloat y  = sfpi::dst_reg[off_y];
        const sfpi::vFloat z  = sfpi::dst_reg[off_z];
        const sfpi::vFloat w  = sfpi::dst_reg[off_w];

        // new_cache = a*x + b*y + c*z + d*w -- three chained tensor*tensor FMAs, entirely in
        // registers (every operand is a per-channel tensor; unlike addcmul's scalar `value`, no
        // term here is uniform across lanes).
        sfpi::vFloat new_cache = wa * x;
        new_cache              = wb * y + new_cache;
        new_cache              = wc * z + new_cache;
        new_cache              = wd * w + new_cache;

        // SiLU(new_cache) = new_cache * sigmoid(new_cache), reusing the existing
        // piecewise-linear sigmoid approximation unchanged.
        sfpi::vFloat sig = _sigmoid_piecewise_linear_positive_(sfpi::abs(new_cache));
        v_if (new_cache < 0.0f)
        {
            sig = 1.0f - sig;
        }
        v_endif;
        const sfpi::vFloat silu_result = new_cache * sig;

        if constexpr (!is_fp32_dest_acc_en)
        {
            sfpi::dst_reg[off_cache_out] = sfpi::convert<sfpi::vFloat16b>(new_cache, sfpi::RoundMode::Nearest);
            sfpi::dst_reg[off_silu_out]  = sfpi::convert<sfpi::vFloat16b>(silu_result, sfpi::RoundMode::Nearest);
        }
        else
        {
            sfpi::dst_reg[off_cache_out] = new_cache;
            sfpi::dst_reg[off_silu_out]  = silu_result;
        }

        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
