// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_exp.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

namespace {
    sfpi::vFloat vConstInfinity;
}

template <bool APPROXIMATION_MODE /*unused*/, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void _calculate_exp2_()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat result;

        // Handle special cases
        v_if (sfpi::isnan(x))
        {
            // NaN input -> NaN output
            result = x;
        }
        v_elseif (sfpi::isinf(x))
        {
            // Check sign of infinity
            sfpi::vInt x_sign = sfpi::exsign(x);
            v_if (sfpi::isequal(x_sign, sfpi::vConst0))  // positive infinity
            {
                result = vConstInfinity;
            }
            v_else  // negative infinity
            {
                result = sfpi::vConst0;  // exp2(-inf) = 0
            }
        }
        v_else
        {
            // Check for overflow and underflow
            // Using thresholds: overflow when x >= 128, underflow when x <= -127
            // These thresholds are for the unbiased exponent of the result (which is x)
            v_if (sfpi::igequal(x, sfpi::vConst(128.0f)))
            {
                // Overflow: result = +infinity
                result = vConstInfinity;
            }
            v_elseif (sfpi::ilequal(x, sfpi::vConst(-127.0f)))
            {
                // Underflow: result = 0.0
                result = sfpi::vConst0;
            }
            v_else
            {
                // Normal case: compute 2^x = 2^(n+f) = 2^n * 2^f, where n = floor(x), f in [0,1)
                // Extract integer and fractional parts
                // We can use floor: n = floor(x), f = x - n
                // We'll use the same rounding function as in exp for consistency, but for floor we can do:
                //   n = trunc(x) for x >= 0, and n = trunc(x) - 1 for x < 0 and not integer
                // However, we can use the SFPI function if available. Let's use trunc and adjust.

                // First, truncate towards zero
                sfpi::vFloat n_trunc = sfpi::trunc(x);
                // Then, if x is negative and not an integer, we need to subtract 1
                sfpi::vFloat is_neg_and_not_int = sfpi::vConst0;
                v_if (sfpi::lessthan(x, sfpi::vConst0))
                {
                    sfpi::vInt x_int = sfpi::reinterpret<sfpi::vInt>(x);
                    sfpi::vFloat x_as_int = sfpi::reinterpret<sfpi::vFloat>(x_int);
                    is_neg_and_not_int = sfpi::iequal(x_as_int, x) ? sfpi::vConst0 : sfpi::vConst1;
                }
                v_endif;
                sfpi::vFloat n = n_trunc - is_neg_and_not_int;
                sfpi::vFloat f = x - n;  // f in [0,1)

                // Compute y = 2^f - 1 using a polynomial approximation
                // We'll use a 5th-order polynomial: y = c1*f + c2*f^2 + c3*f^3 + c4*f^4 + c5*f^5
                // Coefficients from Taylor series of 2^f - 1 at f=0:
                //   c1 = ln(2)
                //   c2 = (ln(2))^2 / 2
                //   c3 = (ln(2))^3 / 6
                //   c4 = (ln(2))^4 / 24
                //   c5 = (ln(2))^5 / 120
                // We'll use the same values as in the exp file for consistency (they use ln(2) = 0.6931471805f)
                constexpr float LN2 = 0.6931471805f;
                constexpr float C1 = LN2;
                constexpr float C2 = LN2 * LN2 * 0.5f;
                constexpr float C3 = LN2 * LN2 * LN2 * (1.0f/6.0f);
                constexpr float C4 = LN2 * LN2 * LN2 * LN2 * (1.0f/24.0f);
                constexpr float C5 = LN2 * LN2 * LN2 * LN2 * LN2 * (1.0f/120.0f);

                // Evaluate polynomial using Horner's method: y = f * (C1 + f * (C2 + f * (C3 + f * (C4 + f * C5))))
                sfpi::vFloat f2 = f * f;
                sfpi::vFloat f3 = f2 * f;
                sfpi::vFloat f4 = f3 * f;
                sfpi::vFloat f5 = f4 * f;

                sfpi::vFloat y = f * (C1 + f * (C2 + f * (C3 + f * (C4 + f * C5))));

                // Now, significand = 1.0 + y = 2^f
                sfpi::vFloat significand = sfpi::vConst1 + y;

                // Exponent (unbiased) = n
                // Biased exponent = n + bias (where bias = 127 for single precision)
                // We'll compute the biased exponent as a float and then convert to int
                sfpi::vFloat biased_exp_float = n + sfpi::vConst(127.0f);
                // Convert to int (bitcast)
                sfpi::vInt biased_exp = sfpi::reinterpret<sfpi::vInt>(biased_exp_float);

                // Build the floating point number: setexp(significand, biased_exp)
                // setexp expects the exponent to be the biased exponent
                result = sfpi::setexp(significand, biased_exp);
            }
        }

        // If the destination is bf16, we need to round the result to bfloat16
        if constexpr (!is_fp32_dest_acc_en)
        {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::NearestEven);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE /*unused*/>
inline void _init_exp2_()
{
    vConstInfinity = sfpi::setexp(sfpi::vConst1, 255);
}

} // namespace ckernel::sfpu