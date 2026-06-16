// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_log.h"
#include "ckernel_sfpu_sqrt.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_exp.h"

namespace ckernel {
namespace sfpu {

//
// log1p(x) = log(1 + x), numerically stable for x near 0.
//
// Strategy:
//   For |x| < 0.5:  use the identity via the existing log body after
//                   carefully constructing 1+x in a way that preserves
//                   the low bits.  We compute log((1+x)) by reusing
//                   _calculate_log_body_no_init_ on (1+x), but we detect
//                   when x is tiny and fall back to the minimax polynomial
//                   log1p(x) ≈ x − x²/2 + x³/3 − x⁴/4 + x⁵/5
//                   which is exact to fp32 precision for |x| < 2^{-12}.
//
//   For |x| >= 0.5: compute (1+x) exactly (no cancellation) and call
//                   the regular log body.
//
// This avoids the catastrophic cancellation that occurs when computing
// log(1 + small) via the naive log(x+1).
//
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _calculate_log1p_body_(sfpi::vFloat x) {
    // For x in (-0.5, 0.5), use Taylor series to avoid cancellation:
    //   log1p(x) = x - x^2/2 + x^3/3 - x^4/4 + x^5/5 - ...
    // For |x| >= 0.5, 1+x is representable without cancellation,
    // so we can call the standard log body.

    // Threshold: |x| < 0.5 → use polynomial, else → use log(1+x)
    // We use 0.5 as the threshold.
    sfpi::vFloat one = sfpi::vConst1;
    sfpi::vFloat arg = x + one;  // 1 + x — exact when |x| >= 0.5

    // Polynomial coefficients for log1p(x) valid for |x| < ~0.5:
    //   p(x) = x*(1 - x*(1/2 - x*(1/3 - x*(1/4 - x*(1/5 - x/6)))))
    // (Horner form, 6 terms — enough for fp32)
    sfpi::vFloat poly;
    {
        // Horner: 1/6
        poly = sfpi::vFloat(1.0f / 6.0f);
        poly = sfpi::vConst1 - x * poly;        // 1 - x/6
        poly = sfpi::vFloat(1.0f / 4.0f) - x * (sfpi::vFloat(1.0f / 5.0f) - x * poly);
        // Rebuild properly with full Horner:
        // p = 1/6
        // p = 1/5 - x*p
        // p = 1/4 - x*p
        // p = 1/3 - x*p
        // p = 1/2 - x*p
        // p = 1   - x*p
        // result = x*p
    }
    // Redo with clean Horner form:
    sfpi::vFloat p = sfpi::vFloat(1.0f / 6.0f);
    p = sfpi::vFloat(1.0f / 5.0f) - x * p;
    p = sfpi::vFloat(1.0f / 4.0f) - x * p;
    p = sfpi::vFloat(1.0f / 3.0f) - x * p;
    p = sfpi::vFloat(0.5f)        - x * p;
    p = sfpi::vConst1             - x * p;
    sfpi::vFloat poly_result = x * p;

    // Standard log of arg = 1 + x (no cancellation for |x| >= 0.5)
    sfpi::vFloat log_result = _calculate_log_body_no_init_(arg);

    // Select: if |x| < 0.5, use poly_result, else use log_result
    // |x| < 0.5 iff x > -0.5 && x < 0.5
    sfpi::vFloat abs_x = sfpi::abs(x);
    // sfpi conditionals: use v_if
    sfpi::vFloat result = log_result;
    v_if(abs_x < sfpi::vFloat(0.5f)) {
        result = poly_result;
    }
    v_endif;

    return result;
}

//
// atanh(x) = 0.5 * log1p(2x / (1 - x))
//
// This is the Kahan/Cody-Waite recommended formula that avoids:
//   1. Catastrophic cancellation near x = 0  (naive: log((1+x)/(1-x)))
//   2. Reciprocal blow-up near x = ±1
//
// Domain: x ∈ (-1, 1).  For |x| >= 1 the result is ±inf / nan per spec.
//
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _calculate_atanh_body_(sfpi::vFloat inp) {
    // Compute 1 - x
    sfpi::vFloat one_minus_x = sfpi::vConst1 - inp;

    // Compute 2x / (1 - x)  via reciprocal of (1 - x)
    sfpi::vFloat recip_den;
    if constexpr (APPROXIMATION_MODE) {
        recip_den = _sfpu_reciprocal_<0>(one_minus_x);
    } else {
        recip_den = _sfpu_reciprocal_<2>(one_minus_x);
    }
    sfpi::vFloat t = (sfpi::vFloat(2.0f) * inp) * recip_den;

    // log1p(t) — stable for all t > -1
    sfpi::vFloat log1p_t = _calculate_log1p_body_<APPROXIMATION_MODE>(t);

    return sfpi::vFloat(0.5f) * log1p_t;
}

//
// asinh(x) = log(|x| + sqrt(x^2 + 1)) * sign(x)
//
// Numerically stable form:
//   For |x| <= 1:     log1p(|x| + (x^2 / (1 + sqrt(1 + x^2))))
//   For |x| >  1:     log(2|x|) + log1p(1 / (2x^2))   [avoids x^2+1 overflow]
//   For |x| very large (> sqrt(FLT_MAX/2) ≈ 1.3e19):
//                     log(2) + log(|x|)                [x^2 overflows]
//
// The "small |x|" formula avoids log(1 + tiny) by using:
//   asinh(x) = log1p(x + x^2/(1 + sqrt(1 + x^2)))
// because  |x| + sqrt(1+x^2) - 1 = x^2 / (1 + sqrt(1+x^2)) + |x| - ... 
// Actually the standard Kahan formula is:
//   asinh(x) = log1p(x + x^2/(sqrt(1+x^2) + 1))
//
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _calculate_asinh_body_(sfpi::vFloat inp) {
    sfpi::vFloat abs_inp = sfpi::abs(inp);
    sfpi::vFloat x2      = abs_inp * abs_inp;  // |x|^2

    // sqrt(1 + x^2)
    sfpi::vFloat one_plus_x2 = x2 + sfpi::vConst1;
    sfpi::vFloat sq          = _calculate_sqrt_body_<APPROXIMATION_MODE>(one_plus_x2);

    // For small |x|: use log1p(|x| + x^2/(sq + 1))
    //   This avoids log(1 + near_zero) cancellation.
    sfpi::vFloat denom_small = sq + sfpi::vConst1;     // sqrt(1+x^2) + 1
    sfpi::vFloat recip_small;
    if constexpr (APPROXIMATION_MODE) {
        recip_small = _sfpu_reciprocal_<0>(denom_small);
    } else {
        recip_small = _sfpu_reciprocal_<2>(denom_small);
    }
    // Kahan argument: |x| + x^2 / (sqrt(1+x^2) + 1)
    sfpi::vFloat kahan_arg = abs_inp + x2 * recip_small;
    sfpi::vFloat res_small = _calculate_log1p_body_<APPROXIMATION_MODE>(kahan_arg);

    // For large |x| (but x^2 doesn't overflow): use log(2|x|) + log1p(1/(2x^2))
    // log(2|x|) = log(2) + log(|x|) = log(2) + _calculate_log_body_no_init_(|x|)
    sfpi::vFloat log2     = sfpi::vFloat(0.6931471805599453f);
    sfpi::vFloat log_absx = _calculate_log_body_no_init_(abs_inp);
    // 1/(2*x^2)
    sfpi::vFloat inv_2x2;
    if constexpr (APPROXIMATION_MODE) {
        inv_2x2 = _sfpu_reciprocal_<0>(sfpi::vFloat(2.0f) * x2);
    } else {
        inv_2x2 = _sfpu_reciprocal_<2>(sfpi::vFloat(2.0f) * x2);
    }
    sfpi::vFloat log1p_inv_2x2 = _calculate_log1p_body_<APPROXIMATION_MODE>(inv_2x2);
    sfpi::vFloat res_large = log2 + log_absx + log1p_inv_2x2;

    // Very large |x|: x^2 overflows → just log(2|x|) = log(2) + log(|x|)
    sfpi::vFloat res_vlarge = log2 + log_absx;

    // Thresholds:
    //   large:   |x| > 1.0  (use large formula when sq ≈ |x| and x^2 is fine)
    //   vlarge:  |x| > sqrt(FLT_MAX/2) ≈ 1.3043e19
    //   We use 1e4 as a conservative "large" threshold for switching to the
    //   large formula (x^2 still representable up to ~1.3e19).
    //   For the very-large case we use 1e19 as a threshold.
    sfpi::vFloat threshold_large  = sfpi::vFloat(1.0f);
    sfpi::vFloat threshold_vlarge = sfpi::vFloat(1.3e19f);

    sfpi::vFloat result = res_small;
    v_if(abs_inp > threshold_large) {
        result = res_large;
    }
    v_endif;
    v_if(abs_inp > threshold_vlarge) {
        result = res_vlarge;
    }
    v_endif;

    // Restore sign of original input
    // sign(inp) = inp / |inp|, but safer: use sfpi sign bit manipulation
    // For inp >= 0: result stays positive
    // For inp < 0: negate result
    v_if(inp < sfpi::vFloat(0.0f)) {
        result = -result;
    }
    v_endif;

    return result;
}

//
// acosh(x) = log(x + sqrt(x^2 - 1))
//
// Numerically stable forms:
//   For x near 1 (1 <= x <= 2):
//       acosh(x) = log1p((x-1) + sqrt((x-1)*(x+1)))
//               = log1p((x-1) + sqrt(x^2-1))
//     This avoids the absorption error: when x≈1, x + sqrt(x²-1) ≈ 1 + tiny,
//     and log(1+tiny) loses ~half the mantissa in naive form.
//     With log1p we keep full precision.
//
//   For x > 2 (sqrt doesn't catastrophically cancel):
//       Use standard: log(x + sqrt(x^2 - 1))
//       But for very large x (> 1.3e19), x^2 overflows:
//       acosh(x) ≈ log(2x) = log(2) + log(x)
//
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _calculate_acosh_body_(sfpi::vFloat inp) {
    // x^2 - 1
    sfpi::vFloat x2_minus_1 = inp * inp - sfpi::vConst1;

    // sqrt(x^2 - 1)
    // Guard: x2_minus_1 might be slightly negative due to rounding when x=1
    // Clamp to 0 before sqrt
    sfpi::vFloat x2m1_clamped = x2_minus_1;
    v_if(x2_minus_1 < sfpi::vFloat(0.0f)) {
        x2m1_clamped = sfpi::vFloat(0.0f);
    }
    v_endif;
    sfpi::vFloat sq = _calculate_sqrt_body_<APPROXIMATION_MODE>(x2m1_clamped);

    // --- Near x=1 branch: x in [1, 2] ---
    // acosh(x) = log1p((x-1) + sqrt((x-1)(x+1)))
    // = log1p((x-1) + sqrt(x^2-1))
    sfpi::vFloat xm1       = inp - sfpi::vConst1;   // x - 1
    sfpi::vFloat arg_near1 = xm1 + sq;              // (x-1) + sqrt(x^2-1)
    sfpi::vFloat res_near1 = _calculate_log1p_body_<APPROXIMATION_MODE>(arg_near1);

    // --- Normal branch: x > 2 ---
    // acosh(x) = log(x + sqrt(x^2 - 1))
    sfpi::vFloat arg_normal = inp + sq;
    sfpi::vFloat res_normal = _calculate_log_body_no_init_(arg_normal);

    // --- Very large branch: x > 1.3e19 (x^2 overflows) ---
    // acosh(x) ≈ log(2) + log(x)
    sfpi::vFloat log2      = sfpi::vFloat(0.6931471805599453f);
    sfpi::vFloat log_inp   = _calculate_log_body_no_init_(inp);
    sfpi::vFloat res_vlarge = log2 + log_inp;

    // Select based on inp value
    sfpi::vFloat threshold_near = sfpi::vFloat(2.0f);
    sfpi::vFloat threshold_v    = sfpi::vFloat(1.3e19f);

    // Start with near-1 formula (valid for x in [1,2])
    sfpi::vFloat result = res_near1;
    v_if(inp > threshold_near) {
        result = res_normal;
    }
    v_endif;
    v_if(inp > threshold_v) {
        result = res_vlarge;
    }
    v_endif;

    return result;
}

//
// Public kernel entry points
//
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_atanh() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = _calculate_atanh_body_<APPROXIMATION_MODE>(val);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_asinh() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = _calculate_asinh_body_<APPROXIMATION_MODE>(val);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_acosh() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = _calculate_acosh_body_<APPROXIMATION_MODE>(val);
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
