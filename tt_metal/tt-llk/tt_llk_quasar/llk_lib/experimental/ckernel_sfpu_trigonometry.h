// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// AI-generated — run_id: 2026-04-02_trigonometry_quasar_e1448d06

#pragma once

#include <cstdint>

#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

// Calculates sine using Maclaurin series with range reduction for SFP_ROWS (2 rows).
// Algorithm: range reduce x to [-pi, pi] via x/pi → round → subtract → scale by pi,
// then apply sin(x) = x - x^3/3! + x^5/5! - x^7/7! [+ x^9/9! - x^11/11!].
// Conditional sign flip for odd half-periods.
template <bool APPROXIMATION_MODE>
inline void _calculate_sine_sfp_rows_()
{
    // 1. Load from Dest
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 0);

    // 2. Range reduction: scale by 1/pi = 0.318309886... (0x3EA2F983)
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, 0x3EA2);
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, 0xF983);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);

    // FP32 → signed INT16 (round to nearest even)
    TTI_SFP_STOCH_RND(p_sfpu::sfp_stochrnd_rnd_mod::NearEven, 0, 0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::sfp_stochrnd_mod::FP32_TO_INT16);

    // Save integer for odd/even test
    TTI_SFPMOV(p_sfpu::LREG2, p_sfpu::LREG3, 0);

    // INT32 (sign-mag) → FP32
    TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG4, 0);

    // Fractional = (x/pi) - round(x/pi), using SFPADD with mod1=2 (negate VC)
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG1, 2);

    // Multiply fractional by pi = 3.14159265... (0x40490FDB) → val in [-pi, pi]
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, 0x4049);
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, 0x0FDB);
    TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);

    // 3. Maclaurin series: sin(x) = x - x^3/3! + x^5/5! - x^7/7! [+ x^9/9! - x^11/11!]
    // LREG1 = val (preserved), LREG0 = power term, LREG5 = output
    TTI_SFPMOV(p_sfpu::LREG1, p_sfpu::LREG5, 0); // output = x
    TTI_SFPMOV(p_sfpu::LREG1, p_sfpu::LREG0, 0); // tmp = x

    // x^3/3!: tmp = tmp*val*val, output += (-1/6)*tmp
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, 0xBE2A); // -1/6 = 0xBE2AAAAB
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, 0xAAAB);
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LREG5, 0);

    // x^5/5!: tmp = tmp*val*val, output += (1/120)*tmp
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, 0x3C08); // 1/120 = 0x3C088888
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, 0x8888);
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LREG5, 0);

    // x^7/7!: tmp = tmp*val*val, output += (-1/5040)*tmp
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, 0xB950); // -1/5040 = 0xB9500CFA
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, 0x0CFA);
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LREG5, 0);

    // Full precision: x^9/9! and x^11/11! terms
    if constexpr (!APPROXIMATION_MODE)
    {
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, 0x3638); // 1/362880 = 0x3638EE91
        TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, 0xEE91);
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LREG5, 0);

        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, 0xB2D7); // -1/39916800 = 0xB2D72D88
        TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, 0x2D88);
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LREG5, 0);
    }

    // 4. Odd/even test + conditional sign flip
    // Load integer mask 0x00000001 into LREG0
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0x0000);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0x0001);

    // AND: LREG0 = LREG0 & LREG3 → isolate bit 0 of saved integer
    TTI_SFPAND(p_sfpu::LREG3, p_sfpu::LREG0);

    // CC test: set flags where LREG0 == 0 (EVEN half-periods), then complement for ODD
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0x6);
    TTI_SFPCOMPC;
    TTI_SFPMOV(p_sfpu::LREG5, p_sfpu::LREG5, 1); // Negate output for ODD lanes (mod1=1)
    TTI_SFPENCC(0, 0);                           // Clear CC result

    // 5. Store result
    TTI_SFPSTORE(p_sfpu::LREG5, 0, ADDR_MOD_7, 0, 0);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void _calculate_sine_(const int iterations)
{
    TTI_SFPENCC(1, 2); // Enable CC mode
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        _calculate_sine_sfp_rows_<APPROXIMATION_MODE>();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();
    }
    TTI_SFPENCC(0, 2); // Disable CC mode
}

// Calculates cosine using Maclaurin series with range reduction for SFP_ROWS (2 rows).
// Algorithm: range reduce x to [-pi, pi] via x/pi -> round -> subtract -> scale by pi,
// then apply cos(x) = 1 - x^2/2! + x^4/4! - x^6/6! [+ x^8/8! - x^10/10!].
// Conditional sign flip for odd half-periods.
template <bool APPROXIMATION_MODE>
inline void _calculate_cosine_sfp_rows_()
{
    // 1. Load from Dest
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 0);

    // 2. Range reduction: scale by 1/pi = 0.318309886... (0x3EA2F983)
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, 0x3EA2);
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, 0xF983);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);

    // FP32 -> signed INT16 (round to nearest even)
    TTI_SFP_STOCH_RND(p_sfpu::sfp_stochrnd_rnd_mod::NearEven, 0, 0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::sfp_stochrnd_mod::FP32_TO_INT16);

    // Save integer for odd/even test
    TTI_SFPMOV(p_sfpu::LREG2, p_sfpu::LREG3, 0);

    // INT32 (sign-mag) -> FP32
    TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG4, 0);

    // Fractional = (x/pi) - round(x/pi), using SFPADD with mod1=2 (negate VC)
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG1, 2);

    // Multiply fractional by pi = 3.14159265... (0x40490FDB) -> val in [-pi, pi]
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, 0x4049);
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, 0x0FDB);
    TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);

    // 3. Maclaurin series: cos(x) = 1 - x^2/2! + x^4/4! - x^6/6! [+ x^8/8! - x^10/10!]
    // LREG1 = val (preserved), LREG0 = power term, LREG5 = output
    TTI_SFPMOV(p_sfpu::LCONST_1, p_sfpu::LREG5, 0);                               // output = 1.0
    TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0); // tmp = x^2

    // x^2/2!: output += (-0.5)*tmp
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, 0xBF00); // -0.5 = 0xBF000000
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, 0x0000);
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LREG5, 0);

    // x^4/4!: tmp = tmp*val*val, output += (1/24)*tmp
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, 0x3D2A); // 1/24 = 0x3D2AAAAB
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, 0xAAAB);
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LREG5, 0);

    // x^6/6!: tmp = tmp*val*val, output += (-1/720)*tmp
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, 0xBAB6); // -1/720 = 0xBAB60B61
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, 0x0B61);
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LREG5, 0);

    // Full precision: x^8/8! and x^10/10! terms
    if constexpr (!APPROXIMATION_MODE)
    {
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, 0x37D0); // 1/40320 = 0x37D00D01
        TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, 0x0D01);
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LREG5, 0);

        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, 0xB493); // -1/3628800 = 0xB493F27E
        TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, 0xF27E);
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LREG5, 0);
    }

    // 4. Odd/even test + conditional sign flip
    // Load integer mask 0x00000001 into LREG0
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0x0000);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0x0001);

    // AND: LREG0 = LREG0 & LREG3 -> isolate bit 0 of saved integer
    TTI_SFPAND(p_sfpu::LREG3, p_sfpu::LREG0);

    // CC test: set flags where LREG0 == 0 (EVEN half-periods), then complement for ODD
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0x6);
    TTI_SFPCOMPC;
    TTI_SFPMOV(p_sfpu::LREG5, p_sfpu::LREG5, 1); // Negate output for ODD lanes (mod1=1)
    TTI_SFPENCC(0, 0);                           // Clear CC result

    // 5. Store result
    TTI_SFPSTORE(p_sfpu::LREG5, 0, ADDR_MOD_7, 0, 0);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void _calculate_cosine_(const int iterations)
{
    TTI_SFPENCC(1, 2); // Enable CC mode
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        _calculate_cosine_sfp_rows_<APPROXIMATION_MODE>();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();
    }
    TTI_SFPENCC(0, 2); // Disable CC mode
}

// Helper: computes ln(x) for x > 0 using exponent extraction + polynomial approximation.
// Convention: input FP32 in LREG0, output ln(x) in LREG5. Trashes LREG1-LREG4.
// Preserves LREG0, LREG6, LREG7.
inline void _calculate_log_body_()
{
    // Step 1: Extract biased exponent via right-shift by 23
    TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG1, 0);         // LREG1 = input copy
    TTI_SFPSHFT(0xFE9, p_sfpu::LREG0, p_sfpu::LREG1, 1); // LREG1 = input >> 23 (logical), mod1=ARG_IMM
    // Mask to 8 bits to remove sign bit that shifted down
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, 0x0000);
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, 0x00FF); // mask = 0x000000FF
    TTI_SFPAND(p_sfpu::LREG2, p_sfpu::LREG1);                       // LREG1 = biased exponent (0-255)

    // Step 2: Unbiased exponent = biased - 127 (two's complement)
    TTI_SFPIADD(0xF81, p_sfpu::LREG1, p_sfpu::LREG1, 5); // LREG1 -= 127, mod1=ARG_IMM|CC_NONE

    // Convert two's complement to sign-magnitude for SFPCAST
    TTI_SFPMOV(p_sfpu::LREG1, p_sfpu::LREG2, 0);       // LREG2 = save two's comp value (for sign)
    TTI_SFPABS(p_sfpu::LREG1, p_sfpu::LREG1, 0);       // LREG1 = |exponent| (2's comp abs, mod1=0)
    TTI_SFPSETSGN(0, p_sfpu::LREG1, p_sfpu::LREG2, 0); // LREG2 = {sign from LREG2} | {mag from LREG1}
    TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG1, 0);      // LREG1 = float(exponent) as signed FP32

    // Step 3: Normalize mantissa to [1,2) by replacing exponent with 127
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, 0x007F);
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, 0xFFFF); // mantissa mask = 0x007FFFFF
    TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG3, 0);                    // LREG3 = input copy
    TTI_SFPAND(p_sfpu::LREG2, p_sfpu::LREG3);                       // LREG3 &= 0x007FFFFF (mantissa bits only)
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, 0x3F80);
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, 0x0000); // exp bias = 0x3F800000
    TTI_SFPOR(p_sfpu::LREG2, p_sfpu::LREG3);                        // LREG3 = x_norm in [1.0, 2.0)

    // Step 4: 3rd-order polynomial: ln(x_norm) = x*(x*(x*A - B) + C) - D
    // A=0.1417 (0x3E111CD0), B=0.8689 (0x3F5E712A), C=2.3099 (0x4013D4E2), D=1.5827 (0x3FCA94C9)
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, 0x3E11);
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, 0x1CD0);               // A
    TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG5, 0); // LREG5 = x*A

    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, 0x3F5E);
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, 0x712A);               // B
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, p_sfpu::LREG5, 2); // LREG5 = x*A - B

    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, 0x4013);
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, 0xD4E2);            // C
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG5, p_sfpu::LREG4, p_sfpu::LREG5, 0); // LREG5 = x*(x*A-B) + C

    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, 0x3FCA);
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, 0x94C9);            // D
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG5, p_sfpu::LREG4, p_sfpu::LREG5, 2); // LREG5 = x*(x*(x*A-B)+C) - D

    // Step 5: Combine: ln(x) = exponent * ln(2) + series_result
    // ln(2) = 0.692871 (0x3F315FFE)
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, 0x3F31);
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, 0x5FFE);
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LREG5, 0); // LREG5 = exp*ln2 + series

    // Step 6: Handle base case: if input == 0, result = -infinity
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0x6);                            // CC set where input == 0
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_UPPER, 0xFF80); // -Inf upper (conditional)
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_LOWER, 0x0000); // -Inf lower (conditional)
    TTI_SFPENCC(0, 0);                                              // Clear CC
    // Result: LREG5 = ln(LREG0)
}

template <bool APPROXIMATION_MODE>
inline void _init_inverse_hyperbolic_()
{
    // No-op on Quasar: sqrt uses hardware SFPNONLINEAR, log body loads constants inline.
}

// Calculates acosh(x) = ln(x + sqrt(x^2 - 1)) with domain checks for SFP_ROWS.
// Uses compute-then-patch: compute general case for all lanes, then overwrite x==1 and x<1 lanes.
template <bool APPROXIMATION_MODE>
inline void _calculate_acosh_sfp_rows_()
{
    // Load input x from Dest
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 0);
    TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG6, 0); // LREG6 = save original input

    // General case (x > 1): acosh(x) = ln(x + sqrt(x^2 - 1))
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);    // LREG1 = x^2
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG1, 2); // LREG1 = x^2 - 1
    TTI_SFPNONLINEAR(p_sfpu::LREG1, p_sfpu::LREG1, p_sfpnonlinear::SQRT_MODE);       // LREG1 = sqrt(x^2 - 1)
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG6, p_sfpu::LREG0, 0);    // LREG0 = sqrt(x^2-1) + x
    _calculate_log_body_();                                                          // LREG5 = ln(sqrt(x^2-1) + x)

    // Patch x == 1 lanes: acosh(1) = 0
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LCONST_1, p_sfpu::LREG0, 2); // LREG0 = x - 1.0
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0x6);                                             // CC where (x - 1.0) == 0
    TTI_SFPMOV(p_sfpu::LCONST_0, p_sfpu::LREG5, 0);                                  // result = 0.0 for x == 1 lanes
    TTI_SFPENCC(0, 0);                                                               // Clear CC

    // Patch x < 1 lanes: acosh(x<1) = NaN
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0x3F80); // 1.0f upper
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0x0000); // 1.0f lower
    TTI_SFPGT(0, p_sfpu::LREG6, p_sfpu::LREG0, 0x1);                // CC set where 1.0 > x (i.e., x < 1.0)
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_UPPER, 0x7FC0); // NaN upper (conditional)
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_LOWER, 0x0000); // NaN lower (conditional)
    TTI_SFPENCC(0, 0);                                              // Clear CC

    // Store result
    TTI_SFPSTORE(p_sfpu::LREG5, 0, ADDR_MOD_7, 0, 0);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void _calculate_acosh_()
{
    TTI_SFPENCC(1, 2); // Enable CC mode
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        _calculate_acosh_sfp_rows_<APPROXIMATION_MODE>();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();
    }
    TTI_SFPENCC(0, 2); // Disable CC mode
}

// Calculates asinh(x) = sign(x) * ln(|x| + sqrt(x^2 + 1)) for SFP_ROWS.
template <bool APPROXIMATION_MODE>
inline void _calculate_asinh_sfp_rows_()
{
    // Load input x from Dest
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 0);
    TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG6, 0); // LREG6 = save original input for sign

    // x^2
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
    // x^2 + 1
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG1, 0);
    // sqrt(x^2 + 1)
    TTI_SFPNONLINEAR(p_sfpu::LREG1, p_sfpu::LREG1, p_sfpnonlinear::SQRT_MODE);
    // |x|
    TTI_SFPABS(p_sfpu::LREG0, p_sfpu::LREG0, 1); // mod1=1 for FP32/sign-magnitude abs
    // sqrt(x^2 + 1) + |x|
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, p_sfpu::LREG0, 0);
    // ln(sqrt(x^2 + 1) + |x|) — input in LREG0, output in LREG5
    _calculate_log_body_();

    // Conditional sign flip: if original x < 0, negate result
    TTI_SFPSETCC(0, p_sfpu::LREG6, 0);           // CC where original x < 0
    TTI_SFPMOV(p_sfpu::LREG5, p_sfpu::LREG5, 1); // Negate result where CC set (mod1=1)
    TTI_SFPENCC(0, 0);                           // Clear CC

    // Store result
    TTI_SFPSTORE(p_sfpu::LREG5, 0, ADDR_MOD_7, 0, 0);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void _calculate_asinh_()
{
    TTI_SFPENCC(1, 2); // Enable CC mode
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        _calculate_asinh_sfp_rows_<APPROXIMATION_MODE>();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();
    }
    TTI_SFPENCC(0, 2); // Disable CC mode
}

// No-op init for atanh: Quasar reciprocal uses hardware SFPNONLINEAR, no setup needed.
template <bool APPROXIMATION_MODE>
inline void _init_atanh_()
{
}

// Calculates atanh(x) = 0.5 * ln((1+x) / (1-x)) with domain checks for SFP_ROWS.
// Uses compute-then-patch: compute general case for all lanes, then patch |x|==1 with +/-Inf
// and |x|>1 with NaN.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _calculate_atanh_sfp_rows_()
{
    // Step 1: Load input and save original x for sign patching at end
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 0);
    TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG6, 0); // LREG6 = original x (preserved across log body)

    // Step 2: Compute |x| and save for domain checks after log body
    TTI_SFPABS(p_sfpu::LREG0, p_sfpu::LREG7, 1); // LREG7 = |x| (mod1=1 for FP32/sign-mag abs)

    // Step 3: Compute general case: 0.5 * ln((1+x) / (1-x))
    // num = 1.0 + x
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG1, 0); // LREG1 = 1.0 + x
    // den = 1.0 - x
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG2, 2); // LREG2 = 1.0 - x (mod1=2)

    // recip_den = 1/(1-x) via hardware approximation
    TTI_SFPNONLINEAR(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpnonlinear::RECIP_MODE); // LREG3 = approx 1/(1-x)

    // Fix sign: SFPSETSGN(0, VC=LREG3, VD=LREG2, 0) -> LREG2 = {sign from LREG2} | {mag from LREG3}
    TTI_SFPSETSGN(0, p_sfpu::LREG3, p_sfpu::LREG2, 0);

    // Optional FP16b truncation for precision in non-FP32 non-approx mode
    if constexpr (!(is_fp32_dest_acc_en || APPROXIMATION_MODE))
    {
        TTI_SFP_STOCH_RND(p_sfpu::sfp_stochrnd_rnd_mod::NearEven, 0, 0, p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::sfp_stochrnd_mod::FP32_TO_FP16B);
    }

    // ratio = (1+x) * recip(1-x) = (1+x) / (1-x)
    TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0); // LREG0 = ratio

    // ln(ratio) via _calculate_log_body_(): input LREG0, output LREG5, trashes LREG1-4
    _calculate_log_body_();

    // result = 0.5 * ln(ratio)
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, 0x3F00);               // 0.5f upper (0x3F000000)
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, 0x0000);               // 0.5f lower
    TTI_SFPMUL(p_sfpu::LREG5, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG5, 0); // LREG5 = 0.5 * ln

    // Step 4: Patch |x| == 1 lanes with +/-Inf
    // Compute 1.0 - |x| and check if zero
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LCONST_1, p_sfpu::LREG7, p_sfpu::LREG0, 2); // LREG0 = 1.0 - |x|
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0x6);                                             // CC where (1.0 - |x|) == 0, i.e., |x| == 1
    // Load +Inf into LREG5 (CC-gated: only affects |x|==1 lanes)
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_UPPER, 0x7F80); // +Inf upper
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_LOWER, 0x0000); // +Inf lower
    // Apply sign from original x: copy x to LREG0, then setsgn(Inf, x) via LREG0
    TTI_SFPMOV(p_sfpu::LREG6, p_sfpu::LREG0, 0);       // LREG0 = original x (CC-gated)
    TTI_SFPSETSGN(0, p_sfpu::LREG5, p_sfpu::LREG0, 0); // LREG0 = {sign(x)} | {mag(+Inf)}
    TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG5, 0);       // LREG5 = signed Inf (CC-gated)
    TTI_SFPENCC(0, 0);                                 // Clear CC

    // Step 5: Patch |x| > 1 lanes with NaN
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0x3F80); // 1.0f upper
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0x0000); // 1.0f lower
    // SFPGT(0, C=LREG0, D=LREG7, 0x1): sets CC where LREG7(|x|) > LREG0(1.0), i.e., |x| > 1
    TTI_SFPGT(0, p_sfpu::LREG0, p_sfpu::LREG7, 0x1);
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_UPPER, 0x7FC0); // NaN upper (conditional)
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_LOWER, 0x0000); // NaN lower (conditional)
    TTI_SFPENCC(0, 0);                                              // Clear CC

    // Step 6: Store result
    TTI_SFPSTORE(p_sfpu::LREG5, 0, ADDR_MOD_7, 0, 0);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void _calculate_atanh_()
{
    TTI_SFPENCC(1, 2); // Enable CC mode
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        _calculate_atanh_sfp_rows_<APPROXIMATION_MODE, is_fp32_dest_acc_en>();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();
    }
    TTI_SFPENCC(0, 2); // Disable CC mode
}

} // namespace sfpu
} // namespace ckernel
