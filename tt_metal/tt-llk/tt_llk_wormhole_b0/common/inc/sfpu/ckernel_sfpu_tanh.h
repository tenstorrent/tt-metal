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
    // Minimax knot placement; see the Wormhole metal copy of this table in
    // hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_tanh.h for the rationale.
    // Measured on n300: overall max |err| 0.144656 -> 0.056339 (2.57x), the new worst point
    // sitting on [0, 1); on [1, 2) it is 0.144656 -> 0.050906.
    sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vUInt(static_cast<std::uint16_t>(0x1AFF)); // 0.8125*x
    sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vUInt(static_cast<std::uint16_t>(0x3814)); // 0.1875*x + 0.625
    sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vUInt(static_cast<std::uint16_t>(0xFF00)); // 1
}

} // namespace sfpu
} // namespace ckernel
