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
template <bool APPROXIMATION_MODE, int WITH_PRECOMPUTED_TANH, int ITERATIONS>
inline void _calculate_tanh_derivative_(const int iterations)
{
    sfpi::vLut16ss s01 = l_reg[sfpi::LRegs::LReg0];
    sfpi::vLut16ss s23 = l_reg[sfpi::LRegs::LReg1];
    sfpi::vLut16ss s45 = l_reg[sfpi::LRegs::LReg2];
    sfpi::vLut16ii i01 = l_reg[sfpi::LRegs::LReg4];
    sfpi::vLut16ii i23 = l_reg[sfpi::LRegs::LReg5];
    sfpi::vLut16ii i45 = l_reg[sfpi::LRegs::LReg6];

    // tanh'(x) = 1 - (tanh(x))^2. SGN_RETAIN makes lut odd, and the square drops the sign.
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[0];

        if constexpr (!WITH_PRECOMPUTED_TANH)
        {
            // The tail pair is (A=0, B=1.0) and the lut evaluates A*|x| + B, so +-inf would
            // come out of it as 0 * inf + 1 = NaN. Move them onto the finite tail first: the
            // lut returns exactly 1.0 there and the result is 0 = sech^2(+-inf), as for every
            // finite |x| >= 3. A NaN input stays NaN. Guarding the input rather than the
            // result because the six table registers leave no LReg to keep it live past the lut.
            //
            // The breakpoints are the lut mode's, not the table's: sfpi::lut's 6-entry default
            // is SFPLUTFP32's TABLE1, whose last split is |x| = 3.0, and the only other 6-entry
            // mode, TABLE2, splits at 4.0, so either mode reads this input from the tail pair.
            // What the value relies on is that pair staying (A=0, B=1.0), which makes the lut
            // exactly 1.0 and 1 - lut^2 exactly 0, and the lut staying in a 6-entry mode. A
            // re-fit of tanh_derivative_init cannot move a breakpoint; one that changes the
            // tail pair breaks this.
            constexpr float SATURATED_TAIL_INPUT = 4.0f;
            v_if (sfpi::is_inf(val))
            {
                val = SATURATED_TAIL_INPUT;
            }
            v_endif;

            val = sfpi::lut(val, s01, i01, s23, i23, s45, i45, sfpi::LutSign::Retain);
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
