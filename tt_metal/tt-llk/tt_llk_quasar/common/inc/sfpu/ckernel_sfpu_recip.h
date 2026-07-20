// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel::sfpu
{
// Computes the reciprocal of a floating point value x.
// max_iter selects the accuracy:
//   0    -> approximate reciprocal only (sfpi::approx_recip, ~7-bit mantissa)
//   1/2  -> Newton-Raphson refinement on top of the approximation (near full precision)
template <int max_iter = 2>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat x)
{
    static_assert(max_iter >= 0 && max_iter <= 2, "max_iter must be between 0 and 2");

    // sfpi::approx_recip(x) will return ±0 for x = ±inf or x ≥ ±2**126, and ±inf for x = ±0.
    sfpi::vFloat y = sfpi::approx_recip(x);

    // Optionally improve the approximation using Newton-Raphson.
    if constexpr (max_iter > 0)
    {
        // Normally, t = 2.0 - x * y, but we negate this (and negate again using y = y * -t later).
        // When x=0 and y=infinity (and vice versa), t=+NaN regardless of the operand signs.
        // Negating the meaning of t makes it easier to detect NaN using a trivial sign check t>=0.
        sfpi::vFloat t = x * y - sfpi::vConstFloatPrgm0;

        if constexpr (max_iter == 2)
        {
            sfpi::vFloat y1 = y * -t - 0.0f;
            // If t=NaN, then t>=0.  This check consumes the SFPNOP slot of the preceding SFPMAD.
            v_if (t < 0)
            {
                t = x * y1 - sfpi::vConstFloatPrgm0;
                y = y1 * -t - 0.0f;
            }
            v_endif;
        }
        else
        {
            // If t=NaN, then t>=0.  This check cannot be hidden in a SFPNOP slot as it depends on the result of the preceding SFPMAD.
            v_if (t < 0)
            {
                y = y * -t - 0.0f;
            }
            v_endif;
        }
    }

    return y;
}

// Programs the SFPU state the reciprocal op relies on:
//   - ADDR_MOD_6: post-increment Dest by one SFPU pass (Quasar writes SFP_ROWS = 2 rows per pass),
//     so the per-pass store advances Dest and the execute loop needs no separate increment.
//   - vConstFloatPrgm0 = 2.0f: the Newton-Raphson constant, only read by the non-approximate path.
template <bool APPROXIMATION_MODE>
inline void _init_reciprocal_()
{
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = ckernel::math::SFP_ROWS},
    }
        .set(ADDR_MOD_6);

    if constexpr (!APPROXIMATION_MODE)
    {
        sfpi::vConstFloatPrgm0 = 2.0f;
    }
}

// Calculates RECIP over a full tile. Quasar exposes exactly two implementations:
//   - approximate reciprocal via the HW nonlinear lookup table (sfpi::approx_recip), and
//   - full-precision reciprocal (Newton-Raphson refinement on top of the LUT seed).
// The LUT is ~1 ULP once the result lands in a bf16 Dest, so the Newton path is only worth
// running for a 32-bit Dest in non-approximate mode; every bf16 case (and any explicit approx
// request) uses the LUT.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_reciprocal_()
{
    constexpr int max_iter = (!is_fp32_dest_acc_en || APPROXIMATION_MODE) ? 0 : 2;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[0]; // load x from dest (SFPLOAD)
        // Store back through ADDR_MOD_6 so the store both writes the result and advances Dest by one pass.
        sfpi::dst_reg[0].mode<>(ckernel::ADDR_MOD_6) = _sfpu_reciprocal_<max_iter>(val);
    }
}

} // namespace ckernel::sfpu
