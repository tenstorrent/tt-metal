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
    constexpr int lut_mode = 0; // SFPLUTFP32_MOD0_FP16_6ENTRY_TABLE1
    sfpi::vUInt l0         = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vUInt l1         = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vUInt l2         = sfpi::l_reg[sfpi::LRegs::LReg2];
    sfpi::vUInt l4         = sfpi::l_reg[sfpi::LRegs::LReg4];
    sfpi::vUInt l5         = sfpi::l_reg[sfpi::LRegs::LReg5];
    sfpi::vUInt l6         = sfpi::l_reg[sfpi::LRegs::LReg6];

#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[0];

        sfpi::dst_reg[0] = lut2(val, l0, l1, l2, l4, l5, l6, lut_mode) + 0.5f;

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
    // imm0 = 0x3DFF;
    // imm1 = 0x21D8;
    // imm2 = 0xFF10;
    // TTI_SFPLOADI(0, 2, imm0);
    // TTI_SFPLOADI(1, 2, imm1);
    // TTI_SFPLOADI(2, 2, imm2);
    // Using a 6 piece LUT to calculate and model sigmoid  directly
    // x <= 0.5 --> 0.2452x + (-0.0004997)
    // x <= 1.0 --> 0.2173x + 0.0152
    // x <= 1.5 --> 0.1731x + 0.05988
    // x <= 2.0 --> 0.1262x + 0.1298
    // x <= 4.0 --> 0.0485x + 0.2998
    // x >  4.0 --> 0.4998

    // imm0[15:0] = A0=0.2452 = 0x33D9 -- imm0[31:16] = A1=0.2173 = 0x32F4
    _sfpu_load_imm32_(0, 0x32F433D9);
    // imm4[15:0] = B0= -0.0004997  = 0x9018 -- imm4[31:16] = B1= 0.0152 = 0x23c8
    _sfpu_load_imm32_(4, 0x23C89018);

    // imm1[15:0] = A2=0.1731 = 0x318a -- imm1[31:16] = A3=0.1262 = 0x300a
    _sfpu_load_imm32_(1, 0x300A318A);
    // imm5[15:0] = B2=0.05988 = 0x2BAA -- imm5[31:16] = B3=0.1298 = 0x3027
    _sfpu_load_imm32_(5, 0x30272BAA);

    // imm2[15:0] = A4=0.0485 = 0x2A35 -- imm2[31:16] = A5=0.0 = 0x7C00
    _sfpu_load_imm32_(2, 0x7C002A35);
    // imm6[15:0] = B4=0.2998 = 0x34CC -- imm6[31:16] = B5=0.4998 = 0x37ff
    _sfpu_load_imm32_(6, 0x37ff34CC);
}

} // namespace sfpu
} // namespace ckernel
