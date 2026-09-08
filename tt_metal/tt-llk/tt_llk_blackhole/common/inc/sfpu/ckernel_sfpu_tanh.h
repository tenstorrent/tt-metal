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
    sfpi::vLut8si si0 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vLut8si si1 = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vLut8si si2 = sfpi::l_reg[sfpi::LRegs::LReg2];

#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[0];
        val              = sfpi::lut(val, si0, si1, si2);
        sfpi::dst_reg[0] = val;

        sfpi::dst_reg++;
    }

    sfpi::l_reg[sfpi::LRegs::LReg0] = si0;
    sfpi::l_reg[sfpi::LRegs::LReg1] = si1;
    sfpi::l_reg[sfpi::LRegs::LReg2] = si2;
}

template <bool APPROXIMATION_MODE>
inline void _init_tanh_()
{
    sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vLut8si(0.90625f, 0.0f);
    sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vLut8si(0.09375f, 0.8125f);
    sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vLut8si(0.0f, 1.0f);
}

} // namespace sfpu
} // namespace ckernel
