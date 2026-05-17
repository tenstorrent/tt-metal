// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
# SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "sfpi.h"
#include "ckernel_sfpu_polyval.h"

namespace ckernel::sfpu
{

/*
 * Optimized exp2 implementation that computes 2^x directly
 * instead of the inefficient exp(x * ln(2)) approach
 */
template <bool APPROXIMATION_MODE /*unused*/, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void _calculate_exp2_()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat result;

        // Handle special cases for exp2(x) = 2^x
        // exp2(+inf) = +inf
        // exp2(-inf) = 0
        // exp2(NaN) = NaN
        // exp2(large positive) -> overflow to +inf
        // exp2(large negative) -> underflow to 0

        // For finite values, compute 2^x directly
        // 2^x = 2^(n + f) = 2^n * 2^f where n = floor(x), f = fraction(x) in [0,1)
        // 2^n can be computed by setting the exponent
        // 2^f for f in [0,1) can be approximated with a polynomial

        if constexpr (is_fp32_dest_acc_en)
        {
            // FP32 accurate path
            // Check for special values first
            sfpi::vBool is_inf = sfpi::isinf(x);
            sfpi::vBool is_neg_inf = sfpi::v_and(sfpi::isinf(x), sfpi::v_lt(x, sfpi::vConst0));
            sfpi::vBool is_nan = sfpi::isnan(x);

            // For normal finite values, compute exp2 directly
            sfpi::vFloat abs_x = sfpi::v_abs(x);
            sfpi::vBool will_overflow = sfpi::v_ge(abs_x, sfpi::vConstF(128.0f)); // 2^128 overflows FP32
            sfpi::vBool will_underflow = sfpi::v_lt(x, sfpi::vConstF(-126.0f)); // 2^-126 underflows to 0 in FP32

            // Start with zero result
            result = sfpi::vConst0;

            // Handle special cases
            sfpi::vBool is_special = sfpi::v_or(sfpi::v_or(is_inf, is_nan), sfpi::v_or(will_overflow, will_underflow));

            // For normal values, compute 2^x = 2^(n+f) = 2^n * 2^f
            sfpi::vBool is_normal = sfpi::v_not(is_special);

            // If normal, compute directly
            sfpi::vFloat n = sfpi::floor(x);           // integer part
            sfpi::vFloat f = sfpi::v_sub(x, n);        // fractional part in [0,1)

            // 2^n = set exponent to n+127 (for FP32)
            // 2^f approximated by polynomial for f in [0,1)
            sfpi::vFloat two_pow_n = sfpi::setexp(sfpi::vConst1, sfpi::v_add(n, sfpi::vConstF(127.0f)));
            sfpi::vFloat two_pow_f = sfpi::polyval_3sf1(f,
                sfpi::vConstF(0.69314718f),  // ln(2)
                sfpi::vConstF(0.24022650f),  // (ln(2)^2)/2!
                sfpi::vConstF(0.05550410f)); // (ln(2)^3)/3!

            result = sfpi::v_mul(two_pow_n, two_pow_f);

            // Special case handling
            result = sfpi::v_sel(is_inf, sfpi::vConstPosInf, result);
            result = sfpi::v_sel(is_neg_inf, sfpi::vConst0, result);
            result = sfpi::v_sel(is_nan, sfpi::vConstQuietNaN, result);
            result = sfpi::v_sel(will_overflow, sfpi::vConstPosInf, result);
            result = sfpi::v_sel(will_underflow, sfpi::vConst0, result);
        }
        else
        {
            // BF16 path - similar optimization applies
            // For BF16, we compute 2^x directly
            sfpi::vBool is_inf = sfpi::isinf(x);
            sfpi::vBool is_neg_inf = sfpi::v_and(sfpi::isinf(x), sfpi::v_lt(x, sfpi::vConst0));
            sfpi::vBool is_nan = sfpi::isnan(x);
            sfpi::vFloat abs_x = sfpi::v_abs(x);
            sfpi::vBool will_overflow = sfpi::v_ge(abs_x, sfpi::vConstF(128.0f)); // Conservative for BF16
            sfpi::vBool will_underflow = sfpi::v_lt(x, sfpi::vConstF(-126.0f));

            // Handle special cases
            sfpi::vBool is_special = sfpi::v_or(sfpi::v_or(is_inf, is_nan), sfpi::v_or(will_overflow, will_underflow));
            sfpi::vBool is_normal = sfpi::v_not(is_special);

            // Start with zero
            result = sfpi::vConst0;

            // If normal, compute 2^x = 2^n * 2^f
            sfpi::vFloat n = sfpi::floor(x);
            sfpi::vFloat f = sfpi::v_sub(x, n);

            // 2^n via exponent setting
            sfpi::vFloat two_pow_n = sfpi::setexp(sfpi::vConst1, sfpi::v_add(n, sfpi::vConstF(127.0f)));
            // 2^f polynomial approximation
            sfpi::vFloat two_pow_f = sfpi::polyval_3sf1(f,
                sfpi::vConstF(0.69314718f),  // ln(2)
                sfpi::vConstF(0.24022650f),  // (ln(2)^2)/2!
                sfpi::vConstF(0.05550410f)); // (ln(2)^3)/3!

            result = sfpi::v_mul(two_pow_n, two_pow_f);

            // Special case handling
            result = sfpi::v_sel(is_inf, sfpi::vConstPosInf, result);
            result = sfpi::v_sel(is_neg_inf, sfpi::vConst0, result);
            result = sfpi::v_sel(is_nan, sfpi::vConstQuietNaN, result);
            result = sfpi::v_sel(will_overflow, sfpi::vConstPosInf, result);
            result = sfpi::v_sel(will_underflow, sfpi::vConst0, result);

            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::NearestEven);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// Initialize the constant for the optimized exp2
// We no longer need to multiply by ln(2) since we compute 2^x directly
template <bool APPROXIMATION_MODE /*unused*/>
inline void _init_exp2_()
{
    // For optimized direct base-2 calculation, we don't need this constant
    // Set to 1.0f for identity multiplication (no effect)
    sfpi::vConstFloatPrgm0 = 1.0f;
}

} // namespace ckernel::sfpu