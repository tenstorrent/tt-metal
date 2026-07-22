// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <limits>

#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

// Full-precision reciprocal square root of a positive floating point value x, ported from the
// Blackhole SQRT_23-bits algorithm (Kokosiński, Z., Gepner, P., Moroz, L. et al. "Fast and accurate
// approximation algorithms for computing floating point square root", Numerical Algorithms (2024),
// https://doi.org/10.1007/s11075-024-01932-7).
//
// Relies on the program constants set by _init_rsqrt_:
//   vConstIntPrgm0   -> initial magic seed (0x5f1110a0)
//   vConstFloatPrgm1 -> 2.2825186f
//   vConstFloatPrgm2 -> 2.2533049f
// Uses only portable sfpi ops (int subtract, fp mul/add, addexp, reinterpret), so no sign-magnitude
// idioms that would misbehave on Quasar.
sfpi_inline sfpi::vFloat _sfpu_rsqrt_body_(const sfpi::vFloat x)
{
    sfpi::vInt i   = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(x) >> 1);
    sfpi::vFloat y = sfpi::as<sfpi::vFloat>(sfpi::vConstIntPrgm0 - i);

    sfpi::vFloat xy          = x * y;
    sfpi::vFloat negative_y  = -y;
    sfpi::vFloat c           = negative_y * xy;
    sfpi::vFloat infinity    = sfpi::sFloat16b(std::numeric_limits<float>::infinity());
    sfpi::vInt infinity_bits = sfpi::as<sfpi::vInt>(infinity);

    y                                = y * (sfpi::vConstFloatPrgm1 + c * (sfpi::vConstFloatPrgm2 + c));
    xy                               = x * y;
    negative_y                       = -y;
    sfpi::vFloat one_minus_xyy       = 1.0f + (negative_y * xy);
    sfpi::vFloat half_y              = sfpi::addexp(y, -1);
    sfpi::vInt x_bits                = sfpi::as<sfpi::vInt>(x);
    sfpi::vInt infinity_minus_x_bits = infinity_bits - x_bits;

    // If x != inf and x != 0, refine with one Newton-Raphson step.
    v_if (infinity_minus_x_bits != 0 && x_bits != 0)
    {
        y = one_minus_xyy * half_y + y;
    }
    // Otherwise, if x = 0 then y = inf; if x = inf then y = 0.
    v_else
    {
        y = sfpi::as<sfpi::vFloat>(infinity_minus_x_bits);
    }
    v_endif;

    v_if (x < 0.0f)
    {
        y = std::numeric_limits<float>::quiet_NaN();
    }
    v_endif;

    return y;
}

// Programs vConstIntPrgm0/vConstFloatPrgm1/vConstFloatPrgm2, the seed and refinement constants read
// only by the full-precision path.
template <bool APPROXIMATION_MODE>
inline void _init_rsqrt_()
{
    if constexpr (!APPROXIMATION_MODE)
    {
        sfpi::vConstIntPrgm0   = 0x5f1110a0;
        sfpi::vConstFloatPrgm1 = 2.2825186f;
        sfpi::vConstFloatPrgm2 = 2.2533049f;
    }
}

// Calculates RSQRT over a full tile. Quasar exposes exactly two implementations:
//   - approximate rsqrt via the HW nonlinear lookup table (approx_recip(approx_sqrt(x))), and
//   - full-precision fp32 rsqrt (_sfpu_rsqrt_body_, ported from Blackhole).
// The LUT is ~1 ULP once the result lands in a bf16 Dest, so the accurate path is only worth
// running for a 32-bit Dest in non-approximate mode; every bf16 case (and any explicit approx
// request) uses the LUT.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_rsqrt_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[0]; // load x from dest (SFPLOAD)

        sfpi::vFloat result;
        if constexpr (!is_fp32_dest_acc_en || APPROXIMATION_MODE)
        {
            result = sfpi::approx_recip(sfpi::approx_sqrt(val));
        }
        else
        {
            result = _sfpu_rsqrt_body_(val);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
