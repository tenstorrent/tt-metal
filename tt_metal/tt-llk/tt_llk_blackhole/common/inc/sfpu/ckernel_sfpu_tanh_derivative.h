// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

// The table comes from tanh_derivative_init in tt-metal's hw/ckernels: slopes in LReg0/1/2
// packed hi/lo, intercepts in LReg4/5/6. It is a 6-entry SFPLUTFP32 FP16 table fitted for
// sech^2 through 1 - lut^2, not the 3-entry SFPLUT this kernel used to read -- the two halves
// must move together, since nothing in the build couples them but this register convention.
//
// The same init loads vConstFloatPrgm0 = FLT_MAX. Before the LUT, |x| > FLT_MAX steps one integer
// down: +-inf becomes +-FLT_MAX, which the (A=0, B=1) tail segment takes to +-1, so tanh'(+-inf)
// is 0 instead of 0 * inf + 1 = NaN, while a NaN stays a NaN (or becomes an infinity, which the
// LUT still turns into NaN). SFPGT compares sign-magnitude, so the mask is -1 exactly there.
template <bool APPROXIMATION_MODE, int WITH_PRECOMPUTED_TANH, int ITERATIONS>
inline void _calculate_tanh_derivative_(const int iterations)
{
    sfpi::vLut16ss s01 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vLut16ss s23 = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vLut16ss s45 = sfpi::l_reg[sfpi::LRegs::LReg2];
    sfpi::vLut16ii i01 = sfpi::l_reg[sfpi::LRegs::LReg4];
    sfpi::vLut16ii i23 = sfpi::l_reg[sfpi::LRegs::LReg5];
    sfpi::vLut16ii i45 = sfpi::l_reg[sfpi::LRegs::LReg6];

    // tanh'(x) = 1 - (tanh(x))^2. SGN_RETAIN makes lut odd, and the square drops the sign.
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[0];

        if constexpr (!WITH_PRECOMPUTED_TANH)
        {
            sfpi::vInt nonfinite = sfpi::vInt(__builtin_rvtt_sfpgt(sfpi::setsgn(val, 0).get(), sfpi::vFloat(sfpi::vConstFloatPrgm0).get(), 8));
            val                  = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(val) + nonfinite);
            val                  = sfpi::lut(val, s01, i01, s23, i23, s45, i45, sfpi::LutSign::Retain);
        }

        val              = val * (-val) + 1.0f;
        sfpi::dst_reg[0] = val;

        sfpi::dst_reg++;
    }

    sfpi::l_reg[sfpi::LRegs::LReg0] = s01;
    sfpi::l_reg[sfpi::LRegs::LReg1] = s23;
    sfpi::l_reg[sfpi::LRegs::LReg2] = s45;
    sfpi::l_reg[sfpi::LRegs::LReg4] = i01;
    sfpi::l_reg[sfpi::LRegs::LReg5] = i23;
    sfpi::l_reg[sfpi::LRegs::LReg6] = i45;
}

} // namespace sfpu
} // namespace ckernel
