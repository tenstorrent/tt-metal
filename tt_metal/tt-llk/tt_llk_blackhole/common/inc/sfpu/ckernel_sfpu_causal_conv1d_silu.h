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
 * @tparam APPROXIMATION_MODE: Retained for signature symmetry with the other SFPU ops; the SiLU
 *   half always evaluates the fast piecewise-linear sigmoid (@ref _sigmoid_piecewise_linear_positive_),
 *   exactly as the shipping @ref _calculate_silu_ does on Blackhole and Wormhole. No separate
 *   higher-accuracy path is offered: the accurate LUT sigmoid (@ref _calculate_sigmoid_) pins
 *   LReg0-2/4-6, which cannot coexist with this op's four-tap tensor FMA inside Blackhole's usable
 *   LReg0-6 budget, and the exp-based `_sfpu_sigmoid_` lives only in the Layer-2 metal stack.
 * @tparam is_fp32_dest_acc_en: When false, both outputs are rounded to bf16 on store (matches
 *   the `calculate_addcmul()`/`calculate_lerp()` convention); when true, both outputs are
 *   stored at full fp32 precision.
 * @tparam ITERATIONS: Number of 32-lane rows to process *within one face* (one row per loop
 *   iteration). A full 32x32 tile is 4 faces of 16x16; the caller must invoke this function once
 *   per face (e.g. via `_llk_math_eltwise_sfpu_apply_vector_mode_(..., VectorMode::RC)`, or an
 *   explicit 4x loop with `_llk_math_eltwise_sfpu_inc_dst_face_addr_()` between calls) to cover
 *   the whole tile -- a single call only ever advances within the current face's window.
 * @param dst_index_wa: Tile holding per-channel weight `a` (multiplies `x`).
 * @param dst_index_wb: Tile holding per-channel weight `b` (multiplies `y`).
 * @param dst_index_wc: Tile holding per-channel weight `c` (multiplies `z`).
 * @param dst_index_wd: Tile holding per-channel weight `d` (multiplies `w`).
 * @param dst_index_x: Newest prior cache entry -- retained by the shift `[new_cache, x, y]`.
 * @param dst_index_y: Middle cache entry -- retained by the shift `[new_cache, x, y]`.
 * @param dst_index_z: Oldest cache entry -- evicted by the shift `[new_cache, x, y]`.
 * @param dst_index_w: Newest sample, already produced by a preceding matmul.
 * @param dst_index_cache_out: Output tile for `new_cache = a*x + b*y + c*z + d*w`. May safely
 *   equal any of `dst_index_wa/wb/wc/wd`: each loop iteration loads one row from every input
 *   operand and consumes it into registers before storing that same row's output, so a weight
 *   slot reused for an output is only overwritten after that row's weight has already been read --
 *   the same `dst_index_out == dst_index_in0` reuse convention
 *   `calculate_addcmul()`/`calculate_where()` already rely on.
 * @param dst_index_silu_out: Output tile for `SiLU(new_cache)`; same reuse rule as
 *   `dst_index_cache_out`, and must differ from it.
 *
 * @note Implements the KDA (Kimi Delta Attention) conv1d node from
 *   tt-blaze#2388/tt-blaze#2429: `new_cache` is the updated causal-conv accumulator (per
 *   channel: `a*x + b*y + c*z + d*w`, all four operands are per-channel BF16 tensors, not
 *   scalars); one output is `SiLU(new_cache)`, the other is the 3-wide cache shift
 *   `[new_cache, x, y]` -- which retains `x`/`y` and evicts the oldest entry `z`. This function
 *   computes only the two values that require new math (`new_cache`, `SiLU(new_cache)`); the
 *   caller re-packs `x`/`y` unchanged to assemble the 3-wide shifted cache, since that half of the
 *   update is a pure data-movement concern, not an SFPU compute one. All operands are expected
 *   already resident in Dest (unpacked/copied by the caller), following the same
 *   `sfpi::dst_reg[dst_index*32 + row]` explicit-addressing idiom as
 *   `calculate_addcmul()`/`calculate_lerp()`; unlike `addcmul`'s scalar `value`, every weight
 *   here is a full per-channel tensor, so all four taps are genuine tensor*tensor FMAs,
 *   accumulated in registers one row at a time (no intermediate Dest round-trips) before that
 *   row's two stores.
 * @note SFPU init contract: like every other SFPU op, the caller must have run the invariant SFPU
 *   init (SFPU config register + zero-increment `ADDR_MOD_7`, e.g.
 *   `_llk_math_eltwise_unary_sfpu_init_once_()`) before this function; the piecewise SiLU path
 *   needs no additional LUT/immediate setup of its own.
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

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        // new_cache = a*x + b*y + c*z + d*w -- four chained tensor*tensor FMAs (every operand is a
        // per-channel tensor; unlike addcmul's scalar `value`, no term here is uniform across
        // lanes). Each operand is read inline into a short-lived vFloat as its FMA consumes it,
        // rather than hoisting all eight inputs into up-front vFloats, so only the accumulator plus
        // the current term's operand pair need be live at once.
        //
        // This ordering also keeps the alias-safety invariant self-evident: all four weights are
        // read (across the four FMAs below) before this row's outputs are stored, so
        // dst_index_cache_out / dst_index_silu_out may reuse a weight slot without ever clobbering
        // a value still needed this iteration. The `#pragma GCC unroll 8` above matches the
        // convention of the other SFPU ITERATIONS loops (where.h, comp.h, ...) and adds no register
        // pressure of its own.
        sfpi::vFloat new_cache = sfpi::vFloat(sfpi::dst_reg[off_wa]) * sfpi::vFloat(sfpi::dst_reg[off_x]);
        new_cache              = sfpi::vFloat(sfpi::dst_reg[off_wb]) * sfpi::vFloat(sfpi::dst_reg[off_y]) + new_cache;
        new_cache              = sfpi::vFloat(sfpi::dst_reg[off_wc]) * sfpi::vFloat(sfpi::dst_reg[off_z]) + new_cache;
        new_cache              = sfpi::vFloat(sfpi::dst_reg[off_wd]) * sfpi::vFloat(sfpi::dst_reg[off_w]) + new_cache;

        // SiLU(new_cache) = new_cache * sigmoid(new_cache), reusing the piecewise-linear sigmoid
        // approximation unchanged.
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
