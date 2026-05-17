// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

// clang-format off: sfpi.h before polyval (polyval uses sfpi_inline); matches blackhole ordering
#include "sfpi.h"
#include "ckernel_sfpu_polyval.h"
// clang-format on
#include "ckernel_sfpu_exp.h"

namespace ckernel::sfpu
{

/*
 * Optimized exp2 for Wormhole B0 SFPU — direct n+f split with minimax polynomial.
 *
 * The previous implementation computed exp2(x) = exp(x * ln(2)), which required a
 * round-trip: multiply by ln(2), then internally multiply by 1/ln(2) to get back to
 * base-2.  This version performs the integer/fractional split directly in base-2,
 * eliminating two floating-point multiplications (one in the caller, one inside exp):
 *
 *   exp2(x) = 2^n * 2^f,   where  x = n + f,  n ∈ ℤ,  f ∈ [0, 1)  or  f ∈ [-½, ½]
 *
 * Two accuracy paths:
 *
 *   • BF16 / 21f  — fast, ~21-bit accurate.  Biases x by +127, converts to int32
 *                    via the IEEE-754 trick (same as the Moroz et al. exp_21f
 *                    algorithm) and applies a degree-2 polynomial correction for 2^f.
 *
 *   • FP32        — high-accuracy, <1 FP32 ULP.  Rounds x to the nearest integer k,
 *                    computes the residual f = x − k, evaluates a degree-7 minimax
 *                    polynomial for 2^f over [−0.5, 0.5], and scales by 2^k via
 *                    exponent injection.
 */

// ─────────────────────────────────────────────────────────────────────────────
//  BF16-accurate exp2  (21-bit accuracy, ≈3 FP32 ULP)
//  Input range:  x ∈ [−127, 128]  (clamped internally to [0, 255] after bias)
// ─────────────────────────────────────────────────────────────────────────────
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp2_21f_bf16_(sfpi::vFloat val)
{
    // Direct bias: x + 127  (no 1/ln2 scaling — input is already in log₂ domain)
    sfpi::vFloat x_biased = val + 127.f;

    // Clamp to [0, 255] so that the int-conversion never wraps.
    sfpi::vFloat threshold_low  = 0.f;
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);
    sfpi::vec_min_max(threshold_low, x_biased);
    sfpi::vec_min_max(x_biased, threshold_high);

    // Convert to int32 via the IEEE-754 exponent/mantissa trick (Moroz et al.).
    // The resulting float bit-pattern encodes both the integer part (in the
    // exponent field) and the fractional part (in the mantissa field).
    sfpi::vInt z = _float_to_int32_for_exp_21f_(x_biased);

    sfpi::vInt exponential_part = sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z), sfpi::ExponentMode::NoDebias);
    sfpi::vInt fractional_part  = sfpi::exman(sfpi::reinterpret<sfpi::vFloat>(z));

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, sfpi::RoundMode::NearestEven);

    // Degree-2 minimax polynomial for 2^f  (same coefficients as the exp 21f
    // path — the polynomial is a correction applied to the fractional mantissa).
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);

    // Recombine:  2^n * 2^f   via exponent injection
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);

    if constexpr (!is_fp32_dest_acc_en)
    {
        y = sfpi::float_to_fp16b(y, sfpi::RoundMode::NearestEven);
    }

    return y;
}

// ─────────────────────────────────────────────────────────────────────────────
//  FP32-accurate exp2  (<1 FP32 ULP)
//  Handles overflow (x ≥ 128), underflow (x ≤ −127), and NaN.
// ─────────────────────────────────────────────────────────────────────────────
sfpi_inline sfpi::vFloat _sfpu_exp2_fp32_accurate_(sfpi::vFloat val)
{
    sfpi::vFloat result = sfpi::vConst0;

    // Thresholds are applied directly to x (already in base-2).
    constexpr float OVERFLOW_THRESHOLD  = 128.0f;
    constexpr float UNDERFLOW_THRESHOLD = -127.0f;

    sfpi::vInt exp_bits = sfpi::exexp(val);

    v_if (val >= OVERFLOW_THRESHOLD)
    {
        result = std::numeric_limits<float>::infinity();
    }
    v_elseif (val <= UNDERFLOW_THRESHOLD)
    {
        result = sfpi::vConst0;
    }
    v_elseif (exp_bits == 255)
    {
        // NaN input — exponent = 255 ⇒ infinity or NaN; the two branches above
        // already caught ±∞, so we know man ≠ 0 here.
        result = std::numeric_limits<float>::quiet_NaN();
    }
    v_else
    {
        // Step 1 — round x to the nearest integer k  (Hacker's Delight method)
        sfpi::vInt k_int;
        sfpi::vFloat k = _sfpu_round_to_nearest_int32_(val, k_int);

        // Step 2 — residual  f = x − k,   f ∈ [−0.5, 0.5]
        sfpi::vFloat f = val - k;

        // Step 3 — degree-7 minimax polynomial for 2^f on [−0.5, 0.5].
        //
        // Derived from  2^f = e^{f·ln2} = Σ (f·ln2)^i / i!
        // Expanded as    2^f ≈ Σ c_i · f^i  where  c_i = ln2^i / i!
        //
        // Degree 7 gives a worst-case error ≈ 5.4 × 10⁻⁹ over [−0.5, 0.5],
        // well within 1 FP32 ULP of the true value.
        sfpi::vFloat p = PolynomialEvaluator::eval(
            f,
            sfpi::vConst1,           // c0 = 1
            0.6931471805599453f,     // c1 = ln2
            0.2402265069591007f,     // c2 = ln2² / 2!
            0.05550073092772832f,    // c3 = ln2³ / 3!
            0.009617899889818306f,   // c4 = ln2⁴ / 4!
            0.001332726284903367f,   // c5 = ln2⁵ / 5!
            0.0001539851334323005f,  // c6 = ln2⁶ / 6!
            0.0000150100981955576f   // c7 = ln2⁷ / 7!
        );

        // Step 4 — scale by 2^k  via exponent injection
        sfpi::vInt p_exp   = sfpi::exexp(p, sfpi::ExponentMode::NoDebias);
        sfpi::vInt new_exp = p_exp + k_int;
        result             = sfpi::setexp(p, new_exp);
    }
    v_endif;

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Top-level microcode loop
// ─────────────────────────────────────────────────────────────────────────────
template <bool APPROXIMATION_MODE /*unused*/, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void _calculate_exp2_()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];

        if constexpr (is_fp32_dest_acc_en)
        {
            sfpi::dst_reg[0] = _sfpu_exp2_fp32_accurate_(v);
        }
        else
        {
            sfpi::vFloat result = _sfpu_exp2_21f_bf16_<true>(v);
            result              = sfpi::float_to_fp16b(result, sfpi::RoundMode::NearestEven);
            sfpi::dst_reg[0]    = result;
        }

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE /*unused*/>
inline void _init_exp2_()
{
    // No constant initialisation required — the direct n+f split avoids the
    // ln(2) constant that the previous implementation stored in vConstFloatPrgm0.
}

} // namespace ckernel::sfpu
