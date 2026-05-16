// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "ckernel_sfpu_exp.h"
#include "sfpi.h"
#include "ckernel_sfpu_polyval.h"

namespace ckernel::sfpu
{

// Helper function to compute floor of a float value
sfpi_inline sfpi::vFloat _sfpu_floor_(sfpi::vFloat val)
{
    // Use round-to-nearest-even then adjust if needed
    sfpi::vInt rounded = sfpi::reinterpret<sfpi::vInt>(val);
    sfpi::vFloat rounded_float = sfpi::reinterpret<sfpi::vFloat>(rounded);

    // If original value is negative and not an integer, we need to subtract 1
    sfpi::vBool is_negative = val < 0;
    sfpi::vBool is_not_integer = val != rounded_float;
    sfpi::vBool needs_adjustment = is_negative & is_not_integer;

    // Subtract 1 if we need to adjust for negative non-integers
    sfpi::vFloat adjustment = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vUInt>(needs_adjustment));
    return rounded_float - adjustment;
}

// Helper function to compute fractional part (x - floor(x))
sfpi_inline sfpi::vFloat _sfpu_fract_(sfpi::vFloat val)
{
    return val - _sfpu_floor_(val);
}

template <bool APPROXIMATION_MODE /*unused*/, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void _calculate_exp2_()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Special case handling
        sfpi::vBool is_inf = sfpi::isinf(x);
        sfpi::vBool is_neg_inf = x < 0 & sfpi::isinf(x);
        sfpi::vBool is_nan = sfpi::isnan(x);

        // For special cases, return IEEE 754 correct values:
        // exp2(+inf) = +inf
        // exp2(-inf) = 0
        // exp2(NaN) = NaN

        // Compute n = floor(x) and f = fract(x)
        sfpi::vFloat n = _sfpu_floor_(x);
        sfpi::vFloat f = _sfpu_fract_(x);

        // Handle special cases first
        sfpi::vFloat result;
        v_if (is_nan)
        {
            result = std::numeric_limits<float>::quiet_NaN();
        }
        v_elseif (is_neg_inf)
        {
            result = 0.0f; // exp2(-inf) = 0
        }
        v_elseif (is_inf)
        {
            result = std::numeric_limits<float>::infinity(); // exp2(+inf) = +inf
        }
        v_else
        {
            // For normal values: exp2(x) = 2^(n+f) = 2^n * 2^f

            // Compute 2^n by setting exponent directly (n is integer)
            // For float32: 2^n = setexp(1.0f, n + 127) where 127 is the bias
            // But we need to handle potential overflow/underflow in exponent

            sfpi::vInt n_int = sfpi::reinterpret<sfpi::vInt>(n);
            sfpi::vInt exp_val = n_int + 127; // Add bias

            // Check for exponent overflow/underflow
            sfpi::vBool exp_overflow = exp_val > 254; // max normal exponent for float32
            sfpi::vBool exp_underflow = exp_val < 0;  // min normal exponent

            // Base value 1.0f
            sfpi::vFloat base = 1.0f;

            // Set exponent to compute 2^n
            sfpi::vFloat pow2n = sfpi::setexp(base, exp_val);

            // Handle special exponent cases:
            // If exponent underflow -> result is 0 (for normal range)
            // If exponent overflow -> result is +inf
            // Otherwise use computed value
            v_if (exp_underflow)
            {
                pow2n = 0.0f;
            }
            v_elseif (exp_overflow)
            {
                pow2n = std::numeric_limits<float>::infinity();
            }

            // Compute 2^f where f ∈ [0, 1) using polynomial approximation
            // For f in [0,1), 2^f ranges from 1 to 2
            // We can use a polynomial approximation or lookup table

            // Using a 3rd degree polynomial approximation for 2^f on [0,1)
            // This gives good accuracy with few operations
            constexpr float C0 = 1.0000000f; // 2^0
            constexpr float C1 = 0.6931472f; // ln(2) * 2^0 (derivative at 0)
            constexpr float C2 = 0.2402265f; // (ln(2)^2)/2 * 2^0
            constexpr float C3 = 0.0555041f; // (ln(2)^3)/6 * 2^0

            sfpi::vFloat pow2f = C0 + f * (C1 + f * (C2 + f * C3));

            // Combine: 2^x = 2^n * 2^f
            result = pow2n * pow2f;

            // Handle special cases where we might have 0 * inf or similar
            v_if (sfpi::isinf(x) & (x > 0))
            {
                result = std::numeric_limits<float>::infinity(); // exp2(+inf) = +inf
            }
            v_elseif (sfpi::isinf(x) & (x < 0))
            {
                result = 0.0f; // exp2(-inf) = 0
            }
        }

        // Handle destination format conversion if needed
        if constexpr (is_fp32_dest_acc_en)
        {
            sfpi::dst_reg[0] = result;
        }
        else
        {
            sfpi::dst_reg[0] = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::NearestEven);
        }

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE /*unused*/>
inline void _init_exp2_()
{
    // No initialization needed for the new implementation
    // The ln(2) constant is no longer used
}

} // namespace ckernel::sfpu