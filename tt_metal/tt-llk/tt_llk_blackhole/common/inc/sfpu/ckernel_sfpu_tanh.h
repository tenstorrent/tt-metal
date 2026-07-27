// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_load_config.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_tanh_(const int iterations)
{
    // SFPU microcode
    sfpi::vUInt l0 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vUInt l1 = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vUInt l2 = sfpi::l_reg[sfpi::LRegs::LReg2];

#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[0];
        val              = lut(val, l0, l1, l2);
        sfpi::dst_reg[0] = val;

        sfpi::dst_reg++;
    }

    sfpi::l_reg[sfpi::LRegs::LReg0] = l0;
    sfpi::l_reg[sfpi::LRegs::LReg1] = l1;
    sfpi::l_reg[sfpi::LRegs::LReg2] = l2;
}

template <bool APPROXIMATION_MODE>
inline void _init_tanh_()
{
    // Load the 3 fp16b LUT coefficients into LReg0-2 via sfpi. Each imm16 is loaded as an
    // unsigned 16-bit value (right-justified, MSBs zeroed), matching the original
    // _sfpu_load_imm16_ -> TT_SFPLOADI mod0==2 path that _calculate_tanh_'s lut() reads back.
    //   imm0 = 0x1DFF -> 0.90625*x
    //   imm1 = 0x481A -> 0.09375*x + 0.8125
    //   imm2 = 0xFF00 -> 1
    sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vUInt(static_cast<std::uint16_t>(0x1DFF));
    sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vUInt(static_cast<std::uint16_t>(0x481A));
    sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vUInt(static_cast<std::uint16_t>(0xFF00));
}

} // namespace sfpu
} // namespace ckernel
