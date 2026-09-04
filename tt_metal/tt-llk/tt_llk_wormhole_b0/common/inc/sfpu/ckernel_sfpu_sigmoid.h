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
    sfpi::vLut16ss s01 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vLut16ss s23 = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vLut16ss s45 = sfpi::l_reg[sfpi::LRegs::LReg2];
    sfpi::vLut16ii i01 = sfpi::l_reg[sfpi::LRegs::LReg4];
    sfpi::vLut16ii i23 = sfpi::l_reg[sfpi::LRegs::LReg5];
    sfpi::vLut16ii i45 = sfpi::l_reg[sfpi::LRegs::LReg6];

#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[0];

        sfpi::dst_reg[0] = sfpi::lut<sfpi::LutMode::Fp16x6_HWM3>(val, s01, i01, s23, i23, s45, i45) + 0.5f;

        sfpi::dst_reg++;
    }

    sfpi::l_reg[sfpi::LRegs::LReg0] = s01;
    sfpi::l_reg[sfpi::LRegs::LReg1] = s23;
    sfpi::l_reg[sfpi::LRegs::LReg2] = s45;
    sfpi::l_reg[sfpi::LRegs::LReg4] = i01;
    sfpi::l_reg[sfpi::LRegs::LReg5] = i23;
    sfpi::l_reg[sfpi::LRegs::LReg6] = i45;
}

template <bool APPROXIMATION_MODE>
inline void _init_sigmoid_()
{
    // Using a 6 piece LUT to calculate and model sigmoid  directly
    sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vLut16ss(0.2452f, 0.2173f);
    sfpi::l_reg[sfpi::LRegs::LReg4] = sfpi::vLut16ss(-0.0004997f, 0.0152);

    sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vLut16ss(0.1731f, 0.1262f);
    sfpi::l_reg[sfpi::LRegs::LReg5] = sfpi::vLut16ii(0.05988f, 0.1298f);

    sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vLut16ss(0.0485f, 0.0f);
    sfpi::l_reg[sfpi::LRegs::LReg6] = sfpi::vLut16ii(0.2998f, 0.4998f);
}

} // namespace sfpu
} // namespace ckernel
