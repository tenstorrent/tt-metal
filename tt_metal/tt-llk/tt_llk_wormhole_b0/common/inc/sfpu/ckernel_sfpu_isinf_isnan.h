// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

// Each predicate is an interval test on the raw fp32 bits, read as a signed integer:
//   isinf    <=> |v| == 0x7F800000        isposinf <=> v == 0x7F800000
//   isnan    <=> |v| >  0x7F800000        isneginf <=> v == 0xFF800000
//   isfinite <=> |v| <= 0x7F7FFFFF
// which is the same predicate as testing the exponent field for all ones and the mantissa
// for zero. The bound is subtracted from in one SFPIADD; for the magnitude tests it cannot
// overflow, and for the equality tests only an all-bits-zero difference matches. The default
// 0.0 is stored first and 1.0 is stored over it on the lanes that satisfy the predicate; that
// store advances Dest through ADDR_MOD_2 (dest.incr = 2, programmed by the op's init).
template <SfpuType operation, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_sfpu_isinf_isnan_()
{
    constexpr std::int32_t bound = operation == SfpuType::isneginf   ? static_cast<std::int32_t>(0xFF800000u)
                                   : operation == SfpuType::isfinite ? 0x7F7FFFFF
                                                                     : 0x7F800000;
    const sfpi::vInt bound_v     = bound;

    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat in  = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = 0.0f;

        if constexpr (operation == SfpuType::isposinf || operation == SfpuType::isneginf)
        {
            v_if (bound_v - sfpi::as<sfpi::vInt>(in) == 0)
            {
                sfpi::dst_reg[0].mode(ADDR_MOD_2) = 1.0f;
            }
            v_endif;
        }
        else
        {
            sfpi::vInt mag = sfpi::as<sfpi::vInt>(sfpi::setsgn(in, 0));
            if constexpr (operation == SfpuType::isinf)
            {
                v_if (bound_v - mag == 0)
                {
                    sfpi::dst_reg[0].mode(ADDR_MOD_2) = 1.0f;
                }
                v_endif;
            }
            else if constexpr (operation == SfpuType::isnan)
            {
                v_if (bound_v < mag)
                {
                    sfpi::dst_reg[0].mode(ADDR_MOD_2) = 1.0f;
                }
                v_endif;
            }
            else if constexpr (operation == SfpuType::isfinite)
            {
                v_if (bound_v >= mag)
                {
                    sfpi::dst_reg[0].mode(ADDR_MOD_2) = 1.0f;
                }
                v_endif;
            }
        }
    }
}

} // namespace ckernel::sfpu
