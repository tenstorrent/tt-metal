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

/**
 * @brief Converts float32 to bfloat16 using IEEE 754 Round-to-Nearest-Even (RNE).
 * Implements the "add 0x7fff + LSB" algorithm for correct tie-breaking, ported
 * from BH. Applied before SFPSTORE because Quasar truncates by default (BH FPU
 * exposes RNE for bf16 store natively).
 *
 * @param in: float32 value to convert
 * @return bf16 value packed in the upper 16 bits of a float32
 */
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

/**
 * @brief LLK caller for binary SFPU operations, currently supports multiply and divide.
 * Vectorized binary divide (in0 / in1) is ported from BH
 * (`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h`,
 * `calculate_sfpu_binary_div`).
 *
 * @note Special cases (matching BH semantics):
 *
 *   - 0 / 0 -> NaN
 *   - x / 0 -> ±inf, sign of x
 *   - x / x -> 1.0 (forced exact, regardless of reciprocal rounding)
 *
 * @tparam APPROXIMATION_MODE: unused, preserved to match the BH metal signature
 * @tparam BINOP: determines which binary op to compute (currently only DIV and MUL supported here)
 * @tparam is_fp32_dest_acc_en: enables FP32 DEST accumulation
 * @param iterations: number of sfpi rows to process (runtime, one call per face)
 * @param dst_index_in0: tile index into DEST for in0
 * @param dst_index_in1: tile index into DEST for in1
 * @param dst_index_out: tile index into DEST for output
 */
template <bool APPROXIMATION_MODE /* unused */, BinaryOp BINOP, bool is_fp32_dest_acc_en = false>
inline void _calculate_sfpu_binary_(
    const int iterations, const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{
    static_assert((BINOP == BinaryOp::DIV || BINOP == BinaryOp::MUL), "This function is only implemented for DIV and MUL");
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr std::uint32_t dst_tile_size_sfpi = 32;
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat in0    = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1    = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat result = 0.0f;

        if constexpr (BINOP == BinaryOp::MUL)
        {
            result = in0 * in1;
        }
        else if constexpr (BINOP == BinaryOp::DIV)
        {
            constexpr int reciprocal_iterations = 2; // Two Newton-Raphson iterations
            result                              = in0 * _sfpu_reciprocal_<reciprocal_iterations>(in1);

            v_if (in1 == 0)
            {
                v_if (in0 == 0)
                {
                    result = std::numeric_limits<float>::quiet_NaN();
                }
                v_else
                {
                    result = std::numeric_limits<float>::infinity();
                    result = sfpi::copysgn(result, in0);
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
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

/**
 * @brief Initialisation hook for binary SFPU kernels.
 * For DIV, programs `sfpi::vConstFloatPrgm0 = 2.0f` for Newton-Raphson reciprocal.
 *
 * @tparam APPROXIMATION_MODE: forwarded to the op-specific init
 * @tparam BINOP: selects which op's init to run
 */
template <bool APPROXIMATION_MODE /*unused*/, BinaryOp BINOP>
inline void _sfpu_binary_init_()
{
    if constexpr (BINOP == BinaryOp::DIV)
    {
        _init_sfpu_reciprocal_<APPROXIMATION_MODE>();
    }
}

} // namespace ckernel::sfpu
