// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_load_config.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_sigmoid_(const int iterations)
{
    sfpi::vFloat16bPair l0 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vFloat16bPair l1 = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vFloat16bPair l2 = sfpi::l_reg[sfpi::LRegs::LReg2];
    sfpi::vFloat16bPair l4 = sfpi::l_reg[sfpi::LRegs::LReg4];
    sfpi::vFloat16bPair l5 = sfpi::l_reg[sfpi::LRegs::LReg5];
    sfpi::vFloat16bPair l6 = sfpi::l_reg[sfpi::LRegs::LReg6];

#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[0];

        sfpi::dst_reg[0] = sfpi::lut<sfpi::LutMode::Fp16x6_HWM3>(val, l0, l1, l2, l4, l5, l6, lut_mode) + 0.5f;

        sfpi::dst_reg++;
    }

    sfpi::l_reg[sfpi::LRegs::LReg0] = l0;
    sfpi::l_reg[sfpi::LRegs::LReg1] = l1;
    sfpi::l_reg[sfpi::LRegs::LReg2] = l2;
    sfpi::l_reg[sfpi::LRegs::LReg4] = l4;
    sfpi::l_reg[sfpi::LRegs::LReg5] = l5;
    sfpi::l_reg[sfpi::LRegs::LReg6] = l6;
}

template <bool APPROXIMATION_MODE>
inline void _init_sigmoid_()
{
    // Using a 6 piece LUT to calculate and model sigmoid  directly
    // x <= 0.5 --> 0.2452x + (-0.0004997)
    // x <= 1.0 --> 0.2173x + 0.0152
    // x <= 1.5 --> 0.1731x + 0.05988
    // x <= 2.0 --> 0.1262x + 0.1298
    // x <= 4.0 --> 0.0485x + 0.2998
    // x >  4.0 --> 0.4998

    // imm0[15:0] = A0=0.2452 = 0x33D9 -- imm0[31:16] = A1=0.2173 = 0x32F4
    sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vFloat16bPair(0.2173f, 0.2452f);
    // imm4[15:0] = B0= -0.0004997  = 0x9018 -- imm4[31:16] = B1= 0.0152 = 0x23c8
    sfpi::l_reg[sfpi::LRegs::LReg4] = sfpi::vFloat16Pair(0.0152, -0.0004997f);

    // imm1[15:0] = A2=0.1731 = 0x318a -- imm1[31:16] = A3=0.1262 = 0x300a
    sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vFloat16Pair(0.1262f, 0.1731f);
    // imm5[15:0] = B2=0.05988 = 0x2BAA -- imm5[31:16] = B3=0.1298 = 0x3027
    sfpi::l_reg[sfpi::LRegs::LReg5] = sfpi::vFloat16Pair(0.1298f, 0.05988f);

    // imm2[15:0] = A4=0.0485 = 0x2A35 -- imm2[31:16] = A5=0.0 = 0x7C00
    sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vFloat16Pair(0.0f, 0.0485f);
    // imm6[15:0] = B4=0.2998 = 0x34CC -- imm6[31:16] = B5=0.4998 = 0x37ff
    sfpi::l_reg[sfpi::LRegs::LReg6] = sfpi::vFloat16Pair(0.4998f, 0.2998f);
}

} // namespace sfpu
} // namespace ckernel
