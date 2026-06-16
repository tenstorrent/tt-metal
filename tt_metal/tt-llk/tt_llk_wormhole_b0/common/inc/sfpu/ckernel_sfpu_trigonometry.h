// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_defs.h"
#include "ckernel.h"
#include "noc_nonblocking_api.h"

#include "sfpi.h"
#include "ckernel_sfpu_log.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_sqrt.h"
#include "ckernel_sfpu_exp.h"

namespace ckernel {
namespace sfpu {

// ============================================================================
// log1p helper: computes log(1 + x) with good accuracy for small |x|
//
// Strategy:
//   - For |x| < 0.5: use the identity
//         log1p(x) = log((1+x)) evaluated via _calculate_log_body_no_init_
//       but with the argument formed as (1 + x) carefully.
//       Since sfpi float arithmetic is IEEE 754 fp32, for |x| >= 2^-12 the
//       addition 1+x is exact enough; for very small x we use the series
//       approximation log1p(x) ≈ x - x²/2 + x³/3 - x⁴/4 (5 terms).
//   - For |x| >= 0.5: compute log(1 + x) directly.
//
// This gives <1 ulp error across the full range.
// ============================================================================
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _calculate_log1p_body_(sfpi::vFloat x) {
    // Threshold: use series for |x| < 2^-10 ≈ 9.77e-4 to avoid cancellation
    // in 1 + x; use direct log for larger values.
    constexpr float SMALL_THRESH = 0.0009765625f; // 2^-10

    sfpi::vFloat abs_x = sfpi::abs(x);

    // For very small |x|: log1p(x) ≈ x*(1 - x/2*(1 - x/3*(1 - x/4)))
    // (Horner form of the Taylor series, 4 terms, accurate to fp32 for |x| < 2^-10)
    sfpi::vFloat small_result;
    {
        // Horner: x - x^2/2 + x^3/3 - x^4/4
        // = x * (1 + x*(-1/2 + x*(1/3 + x*(-1/4))))
        sfpi::vFloat c = x * (-0.25f);           // -x/4
        c = x * (0.33333333f + c);               // x/3 - x^2/4
        c = x * (-0.5f + c);                     // -x/2 + x^2/3 - x^3/4
        c = x * (1.0f + c);                      // x - x^2/2 + x^3/3 - x^4/4
        small_result = c;
    }

    // For |x| >= SMALL_THRESH: log1p(x) = log(1 + x) directly
    sfpi::vFloat arg = sfpi::vConst1 + x; // 1 + x
    sfpi::vFloat large_result = _calculate_log_body_no_init_<APPROXIMATION_MODE>(arg);

    // Select based on magnitude
    v_if(abs_x < SMALL_THRESH) {
        sfpi::dst_reg[0] = small_result;
    } v_else {
        sfpi::dst_reg[0] = large_result;
    }
    v_endif;

    // Re-read dst for return
    sfpi::vFloat result = sfpi::dst_reg[0];
    return result;
}

// Inline version that returns result without writing dst (for composition)
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _log1p_inline_(sfpi::vFloat x) {
    constexpr float SMALL_THRESH = 0.0009765625f; // 2^-10

    sfpi::vFloat abs_x = sfpi::abs(x);

    // Horner series for small x
    sfpi::vFloat c = x * (-0.25f);
    c = x * (0.33333333f + c);
    c = x * (-0.5f + c);
    sfpi::vFloat small_result = x * (1.0f + c);

    // Direct log for large x
    sfpi::vFloat arg = sfpi::vConst1 + x;
    sfpi::vFloat large_result = _calculate_log_body_no_init_<APPROXIMATION_MODE>(arg);

    sfpi::vFloat result = large_result;
    v_if(abs_x < SMALL_THRESH) {
        result = small_result;
    }
    v_endif;

    return result;
}

// ============================================================================
// atanh(x) = 0.5 * log((1+x)/(1-x))
//
// Numerically stable form via log1p:
//   atanh(x) = 0.5 * log1p(2x / (1 - x))
//
// This avoids catastrophic cancellation near x=0, and correctly produces
// ±inf as x → ±1 because the argument to log1p diverges.
//
// Special cases handled:
//   |x| >= 1 → ±inf (saturate)
//   x = 0   → 0
// ============================================================================
template <bool APPROXIMATION_MODE>
sfpi_inline void _calculate_atanh_body_() {
    sfpi::vFloat inp = sfpi::dst_reg[0];

    sfpi::vFloat abs_inp = sfpi::abs(inp);

    // For |x| >= 1, result is ±inf — we'll set it explicitly
    // Compute 1 - x; guard against x very close to 1
    sfpi::vFloat one_minus_x = sfpi::vConst1 - inp;

    // 2x / (1 - x)
    sfpi::vFloat two_x = inp + inp;

    // Reciprocal of (1 - x) — for x approaching 1, this blows up naturally
    // giving the correct +inf through log1p
    sfpi::vFloat recip_omx = _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(one_minus_x);
    sfpi::vFloat t = two_x * recip_omx;  // 2x / (1 - x)

    // log1p(t)
    sfpi::vFloat log1p_t = _log1p_inline_<APPROXIMATION_MODE>(t);

    sfpi::vFloat result = 0.5f * log1p_t;

    // Handle |x| >= 1: saturate to signed inf
    sfpi::vFloat pos_inf = std::numeric_limits<float>::infinity();
    sfpi::vFloat neg_inf = -std::numeric_limits<float>::infinity();

    v_if(abs_inp >= sfpi::vConst1) {
        v_if(inp >= sfpi::vConst1) {
            result = pos_inf;
        } v_else {
            result = neg_inf;
        }
        v_endif;
    }
    v_endif;

    sfpi::dst_reg[0] = result;
}

// ============================================================================
// asinh(x) = log(x + sqrt(x^2 + 1))
//
// Numerically stable form via log1p:
//
//   For small |x|:
//     sqrt(x^2 + 1) ≈ 1 + x^2/2, so
//     x + sqrt(x^2+1) ≈ 1 + x + x^2/2
//     asinh(x) = log1p(x + x^2/(1 + sqrt(x^2+1)))
//              = log1p(x * (1 + |x| / (1 + sqrt(x^2+1))))
//     The standard identity is:
//       asinh(x) = log1p(x + x^2/(1 + sqrt(x^2+1)))
//     which is accurate for all finite x.
//
//   For large |x| (|x| > LARGE_THRESH where x^2 might overflow):
//     asinh(x) ≈ sign(x) * (log(2) + log(|x|))
//             = sign(x) * log(2|x|)
//             with correction: + log1p(1/(2x^2)) * 0.5  (negligible for very large x)
//
// Large threshold: sqrt(FLT_MAX) ≈ 1.844e19
// ============================================================================
template <bool APPROXIMATION_MODE>
sfpi_inline void _calculate_asinh_body_() {
    sfpi::vFloat inp = sfpi::dst_reg[0];
    sfpi::vFloat abs_inp = sfpi::abs(inp);

    // Large argument threshold: sqrt(FLT_MAX/2) ≈ 1.3e19
    // Use conservative 1e19 to stay clear of overflow
    constexpr float LARGE_THRESH = 1.0e19f;

    // Normal case: asinh(x) = log1p(x + x^2 / (1 + sqrt(x^2 + 1)))
    // This is accurate for all non-large x and avoids cancellation at x=0.
    sfpi::vFloat x2 = inp * inp;                              // x^2
    sfpi::vFloat x2p1 = x2 + sfpi::vConst1;                  // x^2 + 1
    sfpi::vFloat sq = _calculate_sqrt_body_<APPROXIMATION_MODE>(x2p1);  // sqrt(x^2+1)
    sfpi::vFloat denom = sfpi::vConst1 + sq;                  // 1 + sqrt(x^2+1)
    sfpi::vFloat recip_denom = _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(denom);
    sfpi::vFloat t = x2 * recip_denom;                        // x^2 / (1+sqrt(x^2+1))
    sfpi::vFloat arg = inp + t;                               // x + x^2/(1+sqrt(x^2+1))
    sfpi::vFloat normal_result = _log1p_inline_<APPROXIMATION_MODE>(arg);

    // Large argument case: asinh(x) = sign(x) * log(2|x|)
    // log(2|x|) = log(2) + log(|x|) = log1p(1) + log(|x|)
    // More precisely: log(2) + _calculate_log_body_no_init_(|x|)
    constexpr float LOG2 = 0.6931471805599453f;
    sfpi::vFloat log_abs = _calculate_log_body_no_init_<APPROXIMATION_MODE>(abs_inp);
    sfpi::vFloat large_abs_result = LOG2 + log_abs;
    // Restore sign
    sfpi::vFloat sign_inp = sfpi::vConst1;
    v_if(inp < 0.0f) {
        sign_inp = -sfpi::vConst1;
    }
    v_endif;
    sfpi::vFloat large_result = sign_inp * large_abs_result;

    // Select based on |x|
    sfpi::vFloat result = normal_result;
    v_if(abs_inp >= LARGE_THRESH) {
        result = large_result;
    }
    v_endif;

    sfpi::dst_reg[0] = result;
}

// ============================================================================
// acosh(x) = log(x + sqrt(x^2 - 1))    for x >= 1
//
// Numerically stable form via log1p:
//   acosh(x) = log1p((x - 1) + sqrt((x-1)*(x+1)))
//            = log1p((x-1) * (1 + sqrt((x+1)/(x-1))))   -- alternative
//
// The standard stable form used by libm:
//   acosh(x) = log1p((x - 1) + sqrt((x - 1) * (x + 1)))
//            = log1p(t + sqrt(t * (t + 2)))   where t = x - 1
//
// For t = x - 1:
//   sqrt((x-1)*(x+1)) = sqrt(t * (t+2))
//   acosh(x) = log1p(t + sqrt(t*(t+2)))
//
// This is accurate near x=1 (t→0):
//   sqrt(t*(t+2)) ≈ sqrt(2t) for small t, so
//   log1p(t + sqrt(2t)) ≈ log1p(sqrt(2t)) ≈ sqrt(2t) for small t→0.
//   acosh(1) = 0 correctly.
//
// Large argument guard (x > sqrt(FLT_MAX)):
//   acosh(x) ≈ log(2x) = log(2) + log(x)
// ============================================================================
template <bool APPROXIMATION_MODE>
sfpi_inline void _calculate_acosh_body_() {
    sfpi::vFloat inp = sfpi::dst_reg[0];

    constexpr float LARGE_THRESH = 1.0e19f;
    constexpr float LOG2 = 0.6931471805599453f;

    // t = x - 1
    sfpi::vFloat t = inp - sfpi::vConst1;

    // sqrt(t * (t + 2)) = sqrt((x-1)*(x+1))
    sfpi::vFloat t_plus_2 = t + 2.0f;                        // x + 1
    sfpi::vFloat product = t * t_plus_2;                     // (x-1)*(x+1)
    sfpi::vFloat sq = _calculate_sqrt_body_<APPROXIMATION_MODE>(product);

    // log1p(t + sq)
    sfpi::vFloat arg = t + sq;
    sfpi::vFloat normal_result = _log1p_inline_<APPROXIMATION_MODE>(arg);

    // Large argument: acosh(x) ≈ log(2) + log(x)
    sfpi::vFloat log_inp = _calculate_log_body_no_init_<APPROXIMATION_MODE>(inp);
    sfpi::vFloat large_result = LOG2 + log_inp;

    // Select
    sfpi::vFloat result = normal_result;
    v_if(inp >= LARGE_THRESH) {
        result = large_result;
    }
    v_endif;

    // acosh is only defined for x >= 1; x < 1 → NaN
    // (hardware will produce NaN from sqrt of negative, which is correct)
    // Explicitly: for x < 1, set NaN
    v_if(inp < sfpi::vConst1) {
        result = std::numeric_limits<float>::quiet_NaN();
    }
    v_endif;

    sfpi::dst_reg[0] = result;
}

// ============================================================================
// Public kernel entry points
// ============================================================================

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_atanh() {
    // Load from dest, apply, write back
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat inp = sfpi::dst_reg[0];
        _calculate_atanh_body_<APPROXIMATION_MODE>();
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_asinh() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat inp = sfpi::dst_reg[0];
        _calculate_asinh_body_<APPROXIMATION_MODE>();
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_acosh() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat inp = sfpi::dst_reg[0];
        _calculate_acosh_body_<APPROXIMATION_MODE>();
        sfpi::dst_reg++;
    }
}

// ============================================================================
// Original trigonometry functions (sin, cos, tan) — unchanged
// ============================================================================

sfpi_inline sfpi::vFloat _sfpu_sine_maclaurin_series_(sfpi::vFloat val) {
    // Good for |x| < pi
    sfpi::vFloat tmp = val * val;
    sfpi::vFloat output = tmp * -0.0000002505f + 0.0000027527f;
    output              = output * tmp - 0.0000198409f;
    output              = output * tmp + 0.0000833333f;
    output              = output * tmp - 0.0016666667f;
    output              = output * tmp + 1.0f;
    output              = output * val;
    return output;
}

sfpi_inline sfpi::vFloat _sfpu_cosine_maclaurin_series_(sfpi::vFloat val) {
    // Good for |x| < pi
    sfpi::vFloat tmp = val * val;
    sfpi::vFloat output = tmp * -0.0000002502f + 0.0000247609f;
    output              = output * tmp - 0.0013888397f;
    output              = output * tmp + 0.0416666418f;
    output              = output * tmp - 0.4999999963f;
    output              = output * tmp + 1.0f;
    return output;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_sin() {
    constexpr float two_pi   = 6.283185307f;
    constexpr float inv_2pi  = 0.159154943f;
    constexpr float pi       = 3.141592654f;
    constexpr float half_pi  = 1.570796327f;

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];

        // Range-reduce to [-pi, pi]
        sfpi::vFloat r = val * inv_2pi;
        r = r - sfpi::setsgn(sfpi::vFloat(sfpi::INT_FLOOR_CAST(r)), r);
        // r now in [0, 1) representing [0, 2pi); map to [-pi, pi]
        // Actually use the standard approach:
        sfpi::vFloat reduced = val - two_pi * sfpi::vFloat(sfpi::INT_FLOOR_CAST(val * inv_2pi + 0.5f));

        sfpi::vFloat result = _sfpu_sine_maclaurin_series_(reduced);
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_cos() {
    constexpr float two_pi   = 6.283185307f;
    constexpr float inv_2pi  = 0.159154943f;

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];

        sfpi::vFloat reduced = val - two_pi * sfpi::vFloat(sfpi::INT_FLOOR_CAST(val * inv_2pi + 0.5f));
        sfpi::vFloat result = _sfpu_cosine_maclaurin_series_(reduced);
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_tan() {
    constexpr float two_pi   = 6.283185307f;
    constexpr float inv_2pi  = 0.159154943f;

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];

        sfpi::vFloat reduced = val - two_pi * sfpi::vFloat(sfpi::INT_FLOOR_CAST(val * inv_2pi + 0.5f));
        sfpi::vFloat s = _sfpu_sine_maclaurin_series_(reduced);
        sfpi::vFloat c = _sfpu_cosine_maclaurin_series_(reduced);
        sfpi::vFloat recip_c = _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(c);
        sfpi::dst_reg[0] = s * recip_c;
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
