// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// clang-format off: sfpi_inline must be defined before ckernel_sfpu_polyval.h
#include "sfpi.h"

#include "ckernel_sfpu_polyval.h"
// clang-format on

namespace ckernel::sfpu
{

/*
 * _float_to_int32_positive_ uses branches to handle special cases.
 * With exp21f function, some of these cases never happen (e.g. negative
 * exponent, overflow), allowing for a branch free (and much smaller)
 * algorithm to compute the integer value.
 *
 * The constraint on `val` is: 0 <= val < 128.0f
 * Note: Unlike _float_to_int32_positive_, this function assumes that the
 * value has been divided by 2^23. The output value will be scaled by 2^23
 * compared to `val`. If that was not the case, we would have had to shift
 * by `exp - 23` instead of `exp`. This saves 1 SFPADDI instruction.
 */
sfpi_inline sfpi::vInt _float_to_int32_for_exp_21f_(sfpi::vFloat val)
{
    sfpi::vInt exp = sfpi::exexp(val);
    sfpi::vInt man = sfpi::exman8(val); // mantissa with implicit bit (man in [1; 2])
    man            = sfpi::reinterpret<sfpi::vInt>(sfpi::shft(sfpi::reinterpret<sfpi::vUInt>(man), exp));
    return man;
}

/*
 * Polynomial-approximation implementation of the exponential function based on
 * "Simple Multiple Precision Algorithms for Exponential Functions [Tips & Tricks]"
 * by Moroz et al. 2022 (https://doi.org/10.1109/MSP.2022.3157460), specifically
 * the `exp_21f` algorithm described in Section 5.
 *
 * @param val Input value (sfpi::vFloat vector); any finite floating-point value.
 * @return sfpi::vFloat result of exp(val).
 *
 * Ported from BH (`tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h`) so the
 * BH-style vectorized ELU implementation can compute exp(x) without depending
 * on the Quasar HW SFPNONLINEAR EXP_MODE path.
 */
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_21f_bf16_(sfpi::vFloat val)
{
    // exp(x) = 2**(x / ln2) = 2**(z_i) * 2**(z_f)
    //   z_i = trunc(x / ln2)               (integer part)
    //   z_f = x/ln2 - trunc(x/ln2)         (fractional part)
    //
    // Per the paper (Sections 2/3):
    //   z = bias + x * factor * N_m, with factor = log(2) * 2^23, bias = 127 * 2^23
    //   i.e. z = val * log(2) * 2^23 + 127 * 2^23
    //
    // The multiplication by 2^23 is done implicitly inside _float_to_int32_for_exp_21f_.
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2      = (val * ONE_LN2 + 127.f);

    // Intermediary values overflow if xlog2 is outside of [0, 256[, leading to
    // invalid results instead of 0 (when input < -88.5) and +inf (when input > 88.5).
    // Clamp xlog2 to [0, 255]. (Threshold values are rounded to bf16; this does
    // not change the result but only requires one SFPLOADI vs. two.)
    sfpi::vFloat threshold_low  = 0.f;
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);
    sfpi::vec_min_max(threshold_low, xlog2);
    sfpi::vec_min_max(xlog2, threshold_high);

    sfpi::vInt z = _float_to_int32_for_exp_21f_(xlog2);

    sfpi::vInt exponential_part = exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z)); // 2**(integer part of val/ln2)
    sfpi::vInt fractional_part  = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));   // leftover, in [0; 1]

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, sfpi::RoundMode::NearestEven);

    // Refine 2**(x_f) on [0; 2^23] with a 2nd-degree polynomial.
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);

    // Recombined exponent and mantissa: equivalent to 2**(x_i) * 2**(x_f).
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);

    if constexpr (!is_fp32_dest_acc_en)
    {
        // LRegs work on float32 data; if Dest is bfloat16 then SFPSTORE truncates,
        // which can reduce accuracy (e.g. 9**2 = 80.8 rounds to 80.5 rather than 81).
        // Explicitly convert to bfloat16 using round-to-nearest-even to avoid this.
        y = sfpi::float_to_fp16b(y, sfpi::RoundMode::NearestEven);
    }

    return y;
}

} // namespace ckernel::sfpu
