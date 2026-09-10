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

// Approximation-mode only: this is the 3-segment SFPLUT and there is no accurate path here.
// Metal's calculate_tanh (hw/ckernels/<arch>/metal/llk_api/llk_sfpu/ckernel_sfpu_tanh.h) is the
// one that branches on APPROXIMATION_MODE; reach for that if you need the accurate tanh.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_tanh_(const int iterations)
{
    static_assert(APPROXIMATION_MODE, "_calculate_tanh_ implements only the approximate (SFPLUT) tanh");

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
    static_assert(APPROXIMATION_MODE, "_init_tanh_ loads only the approximate (SFPLUT) table");

    // Segments are |x| buckets split at exactly 1.0 and 2.0; SGN_RETAIN gives
    // sign(x) * (A*|x| + B). Remez minimax fit holding tanh(0) = 0, continuity at |x| = 1 and
    // exact 1.0 saturation from |x| = 2, which leave one free parameter (LReg0's slope).
    // Max abs error 0.0563, against 0.1447 for the previous table. Derivation and the
    // coefficient byte encoding are in APPROX_TANH_RETUNE.md.
    sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vLut8si(0.8125f, 0.0f);
    sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vLut8si(0.1875f, 0.625f);
    sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vLut8si(0.0f, 1.0f);
}

} // namespace sfpu
} // namespace ckernel
