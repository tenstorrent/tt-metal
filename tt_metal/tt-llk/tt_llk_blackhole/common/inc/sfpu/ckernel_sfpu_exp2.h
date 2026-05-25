// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "sfpi.h"

namespace ckernel::sfpu
{

/**
 * Optimized exp2(x) = 2^x implementation for Wormhole B0 / Blackhole.
 *
 * This implementation bypasses the exp(x * ln(2)) round-trip to reduce cycle count
 * and improve accuracy (target <= 1 ulp).
 *
 * Algorithm:
 * 1. Split x = n + f where n = floor(x) and f = x - n.
 * 2. Compute 2^f using a degree-5 minimax polynomial for f in [0, 1).
 * 3. Add n to the IEEE-754 exponent of 2^f to get the final result.
 */
template <bool APPROXIMATION_MODE /*unused*/, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void _calculate_exp2_()
{
    // Minimax polynomial coefficients for 2^f on [0, 1]
    const float c1 = 0.6931471806f;
    const float c2 = 0.2402265070f;
    const float c3 = 0.0555041087f;
    const float c4 = 0.0096181291f;
    const float c5 = 0.0013333558f;

    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat x = sfpi::dst_reg[0];
        
        // Handle special cases: +/- inf and NaN
        sfpi::vBool is_nan = sfpi::isnan(x);
        sfpi::vBool is_pos_inf = (x == sfpi::vConstInf);
        sfpi::vBool is_neg_inf = (x == sfpi::vConstNegInf);

        // n = floor(x), f = x - n
        sfpi::vFloat n = sfpi::floor(x);
        sfpi::vFloat f = x - n;

        // Degree 5 polynomial approximation for 2^f using Horner's method
        // Restructured to use a * b + c pattern for single-instruction MADD mapping
        // res = ((((c5 * f + c4) * f + c3) * f + c2) * f + c1) * f + 1.0
        sfpi::vFloat poly = c5 * f + c4;
        poly = poly * f + c3;
        poly = poly * f + c2;
        poly = poly * f + c1;
        sfpi::vFloat res = poly * f + 1.0f;

        // Add n to the exponent
        // setexp internally handles the bias and packing
        res = sfpi::setexp(res, n);

        // Branchless special case handling
        res = sfpi::select(is_pos_inf, sfpi::vConstInf, res);
        res = sfpi::select(is_neg_inf, 0.0f, res);
        res = sfpi::select(is_nan, sfpi::vConstNaN, res);

        sfpi::dst_reg[0] = res;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE /*unused*/>
inline void _init_exp2_()
{
    // No initialization needed for the direct implementation
}

} // namespace ckernel::sfpu
