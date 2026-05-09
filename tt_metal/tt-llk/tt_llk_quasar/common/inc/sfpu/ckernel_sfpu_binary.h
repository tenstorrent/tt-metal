// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "ckernel_sfpu_recip.h"
#include "llk_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

// Convert float32 to bfloat16 using IEEE 754 Round-to-Nearest-Even (RNE).
// Implements the "add 0x7fff + LSB" algorithm for correct tie-breaking. Ported
// verbatim from BH (`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/
// ckernel_sfpu_binary.h`) so that the bf16 output of the vectorized div helper
// below matches the FPU's RNE rounding behaviour (the BH FPU exposes RNE for
// bf16 store; SFPSTORE on Quasar truncates by default, so we apply the
// software RNE conversion before the final store when Dest is in bf16 mode).
sfpi_inline sfpi::vFloat _float32_to_bf16_rne_(sfpi::vFloat in)
{
    sfpi::vUInt bits = sfpi::reinterpret<sfpi::vUInt>(in);

    // Extract the LSB of what will become the bf16 mantissa (bit 16 of float32).
    // Needed for the tie-breaker: round to even.
    sfpi::vUInt lsb = (bits >> 16) & 1;

    // Add 0x7fff + lsb to implement RNE:
    // - lower 16 bits > 0x8000      → overflow, rounds up
    // - lower 16 bits < 0x8000      → no overflow, rounds down
    // - lower 16 bits == 0x8000 (tie)
    //     and lsb=0: 0x7fff+0=0xffff, no overflow → stays even
    //     and lsb=1: 0x7fff+1=0x8000,    overflow → rounds up to even
    bits = bits + 0x7fffU + lsb;

    // Clear the lower 16 bits to get bf16 in upper 16 bits (bf16 format in float32).
    bits = bits & 0xFFFF0000U;

    return sfpi::reinterpret<sfpi::vFloat>(bits);
}

// Vectorized binary divide (in0 / in1), ported from BH
// (`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h`,
// `calculate_sfpu_binary_div`).
//
// Special cases (matching the BH semantics):
//     0   /   0  -> NaN
//     x   /   0  -> ±inf, sign of x
//     x   /   x  -> 1.0 (forced exact, regardless of reciprocal rounding)
//
// `iterations` is a runtime parameter on Quasar (matches the
// `_calculate_*_(const int iterations, ...)` convention used by the existing
// Quasar SFPU kernels and `_llk_math_sfpu_params_` / `_llk_math_eltwise_binary_
// sfpu_params_`, which call this helper once per face).
//
// `dst_index_in0` / `dst_index_in1` / `dst_index_out` are tile indices into
// DEST; `dst_tile_size_sfpi = 32` is the per-tile sfpi-row stride (a tile
// occupies 64 hw rows in DEST when accessed via sfpi load/store, and
// SFP_DESTREG_STRIDE = 2). The advance through a face is performed with
// `sfpi::dst_reg++` inside the inner loop; the per-face dst-counter advance
// between calls is handled by the caller (e.g. via
// `_llk_math_eltwise_binary_sfpu_inc_dst_face_addr_()`).
//
// Note: BINOP is preserved as a template parameter to match the BH metal
// signature (which lets the same wrapper template fan out to all binary ops),
// but it is unused in the body — this helper is the dedicated DIV path.
template <bool APPROXIMATION_MODE, BinaryOp BINOP, bool is_fp32_dest_acc_en = false>
inline void _calculate_sfpu_binary_div_(
    const int iterations, const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr std::uint32_t dst_tile_size_sfpi = 32;
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat in0    = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1    = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat result = in0 * _sfpu_reciprocal_<2>(in1);

        v_if (in1 == 0)
        {
            v_if (in0 == 0)
            {
                result = std::numeric_limits<float>::quiet_NaN();
            }
            v_else
            {
                result = std::numeric_limits<float>::infinity();
                result = sfpi::setsgn(result, in0);
            }
            v_endif;
        }
        v_elseif (in0 == in1)
        {
            result = sfpi::vConst1;
        }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en)
        {
            // Software RNE conversion to match FPU bf16 rounding (Quasar SFPSTORE
            // truncates by default).
            result = _float32_to_bf16_rne_(result);
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

// Initialisation hook for binary SFPU kernels.
// For DIV, programs `sfpi::vConstFloatPrgm0 = 2.0f` so the reciprocal helper's
// Newton-Raphson refinement uses the correct constant. Other binops added in
// the future (RSUB / POW / XLOGY ...) can extend this dispatch the same way
// the BH `_sfpu_binary_init_` does.
template <bool APPROXIMATION_MODE /*unused*/, BinaryOp BINOP>
inline void _sfpu_binary_init_()
{
    if constexpr (BINOP == BinaryOp::DIV)
    {
        _init_sfpu_reciprocal_<APPROXIMATION_MODE>();
    }
}

} // namespace ckernel::sfpu
