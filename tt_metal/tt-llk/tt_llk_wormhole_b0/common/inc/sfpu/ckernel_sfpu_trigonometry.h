// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "noc_nonblocking_api.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_log.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_sqrt.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// ---------------------------------------------------------------------------
// Internal helper: log1p(x) = log(1 + x)
//
// Uses the identity:
//   log1p(x) = log((1+x)/1) decomposed as:
//     if |x| < threshold: use direct _calculate_log_body_no_init_(1+x)
//     but to avoid cancellation we use:
//       log1p(x) = x - x^2/2 + x^3/3 - ...   for |x| < ~2^-11  (not used here)
//
// We use a split approach:
//   For |x| <= 0.5:
//     log1p(x) = log(1+x)   computed as log applied to (1+x), but using the
//     compensated form:  let s = x/(2+x), then
//       log1p(x) = 2*s + 2/3*s^3 + 2/5*s^5 + ...  (Halley series)
//     This is the standard libm approach.
//
//   For |x| > 0.5:
//     log1p(x) = _calculate_log_body_no_init_(1 + x)  (no cancellation risk)
//
// For the SFPU, we implement a practical version:
//   - The hardware log is accurate to ~1 ULP for arguments well away from 1.
//   - Near 1 (i.e., small x), we use the series via s = x/(2+x).
// ---------------------------------------------------------------------------

// Minimax polynomial for log1p via s = x/(2+x), log1p(x) = 2*(s + s^3/3 + s^5/5 + ...)
// We use 4 terms: 2s*(1 + s^2*(1/3 + s^2*(1/5 + s^2/7)))
// This is accurate to ~2^-24 for |x| <= 0.5
template <bool APPROXIMATION_MODE>
sfpi_inline vFloat _log1p_(vFloat x) {
    // For |x| > 0.5: direct log(1+x) is fine (no catastrophic cancellation)
    // For |x| <= 0.5: use series via s = x/(2+x)

    vFloat one_plus_x = vConst1 + x;

    // Series approach: s = x / (x + 2)
    // log1p(x) = 2*s*(1 + s^2*(1/3 + s^2*(1/5 + s^2*(1/7))))
    // This avoids cancellation for small x.
    //
    // For large |x|, we fall through to direct log(1+x).
    //
    // We implement a blended approach: always compute the series for |x| <= 0.5,
    // and use direct log otherwise. On SFPU, conditional execution uses v_if.

    vFloat result;

    // Compute s = x / (2 + x)
    // = x * recip(2 + x)
    vFloat two_plus_x = x + 2.0f;
    vFloat inv_two_plus_x = _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(two_plus_x);
    vFloat s = x * inv_two_plus_x;

    // Horner's method for 1/3 + s^2*(1/5 + s^2*(1/7 + s^2/9))
    vFloat s2 = s * s;
    // 4 terms: coefficients 1, 1/3, 1/5, 1/7
    // poly = 1 + s2*(1/3 + s2*(1/5 + s2*(1/7)))
    vFloat poly = s2 * (1.0f / 7.0f);
    poly = s2 * (poly + (1.0f / 5.0f));
    poly = s2 * (poly + (1.0f / 3.0f));
    poly = poly + 1.0f;
    vFloat series_result = 2.0f * s * poly;

    // Direct log(1+x) for |x| > 0.5
    vFloat direct_result = _calculate_log_body_no_init_(one_plus_x);

    // Blend: use series when |x| <= 0.5
    // abs(x) <= 0.5  iff  x >= -0.5 && x <= 0.5
    result = direct_result;
    v_if(x >= -0.5f && x < 0.5f) {
        result = series_result;
    }
    v_endif;

    return result;
}

// ---------------------------------------------------------------------------
// atanh(x) = 0.5 * log1p(2x / (1 - x))
//
// This avoids:
//   - Catastrophic cancellation in log((1+x)/(1-x)) for x → 0
//   - Reciprocal blowup at x → ±1 before the log can respond
//
// Domain: |x| < 1. For |x| >= 1: atanh(±1) = ±inf, atanh(|x|>1) = NaN
// ---------------------------------------------------------------------------
template <bool APPROXIMATION_MODE>
sfpi_inline vFloat _calculate_atanh_body_(vFloat inp) {
    // Compute 2x / (1 - x)
    // = 2x * recip(1 - x)
    vFloat one_minus_x = vConst1 - inp;

    // Guard against one_minus_x = 0 (x = 1): recip(0) = inf, log1p(inf) = inf, *0.5 = inf. OK.
    vFloat inv_one_minus_x = _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(one_minus_x);
    vFloat arg = 2.0f * inp * inv_one_minus_x;

    // atanh(x) = 0.5 * log1p(arg)
    vFloat result = 0.5f * _log1p_<APPROXIMATION_MODE>(arg);

    return result;
}

// ---------------------------------------------------------------------------
// asinh(x) = log(|x| + sqrt(x^2 + 1))
//
// Numerically stable via log1p:
//   asinh(x) = sign(x) * log1p(|x| + (sqrt(x^2+1) - 1))
//            = sign(x) * log1p(|x| + x^2 / (sqrt(x^2+1) + 1))
//
// The second form avoids cancellation when x is small:
//   sqrt(x^2+1) - 1 ≈ x^2/2 for small x, so arg to log1p is ≈ |x| + x^2/2 → 0
//
// For large |x| (x^2 might overflow fp32, threshold ~1.84e19):
//   asinh(x) ≈ sign(x) * (log(2) + log(|x|))   = sign(x) * log1p(2|x| - 1)  ... not quite
//   Use: asinh(x) = sign(x) * log(2*|x|) = sign(x) * (log(2) + log(|x|))
//   More precisely: sign(x) * log1p(2|x| + 1/(2|x|) - 1) is still tricky.
//   We use: for |x| > LARGE_THRESH, result = sign(x)*(log(|x|) + log(2))
//           via _calculate_log_body_no_init_(abs_x) + log2_recip ... 
//   Actually simplest: sign(x) * _calculate_log_body_no_init_(2*abs_x)
//   since log(2|x|) = log(2) + log(|x|), and for |x| >> 1, asinh(x) ≈ log(2|x|).
// ---------------------------------------------------------------------------
template <bool APPROXIMATION_MODE>
sfpi_inline vFloat _calculate_asinh_body_(vFloat inp) {
    // abs(x)
    vFloat abs_x = sfpi::abs(inp);

    // Compute x^2
    vFloat x2 = inp * inp;

    // sqrt(x^2 + 1)
    vFloat sqrt_x2p1 = _calculate_sqrt_body_<APPROXIMATION_MODE>(x2 + vConst1);

    // Numerically stable argument for log1p:
    // arg = abs_x + x^2 / (sqrt_x^2+1 + 1)
    // This equals sqrt(x^2+1) - 1 + abs_x, but computed stably.
    vFloat sqrt_p1_plus1 = sqrt_x2p1 + vConst1;
    vFloat inv_sqrt_p1_plus1 = _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(sqrt_p1_plus1);
    vFloat correction = x2 * inv_sqrt_p1_plus1;
    vFloat arg = abs_x + correction;

    // asinh(x) = sign(x) * log1p(arg)
    vFloat log_val = _log1p_<APPROXIMATION_MODE>(arg);

    // Restore sign
    vFloat result = log_val;
    v_if(inp < 0.0f) {
        result = -log_val;
    }
    v_endif;

    // Handle large |x| to avoid overflow in x^2:
    // When |x| > large_thresh (~1e19 for fp32, but fp32 max is ~3.4e38)
    // x^2 overflows when |x| > sqrt(FLT_MAX) ≈ 1.844e19
    // For such x, use asinh(x) ≈ sign(x) * log(2*|x|)
    // log(2*|x|) = _calculate_log_body_no_init_(2*abs_x) if 2*abs_x doesn't overflow
    // Since abs_x > 1.844e19, 2*abs_x could overflow if abs_x > 1.7e38
    // But log(2*abs_x) = log(abs_x) + log(2), so use log(abs_x) + 0.6931472f
    // We detect overflow by checking x2 == +inf (after x2 = inp*inp).
    // However, once x2 is computed and overflowed, we need a separate path.
    // Recompute for large x:
    vFloat large_result;
    {
        // log(abs_x) + log(2)
        vFloat log_abs_x = _calculate_log_body_no_init_(abs_x);
        large_result = log_abs_x + 0.6931471805599453f;  // log(2)
        v_if(inp < 0.0f) {
            large_result = -large_result;
        }
        v_endif;
    }

    // Select large path when x^2 overflowed (i.e., x2 is +inf)
    // We use: x2 > 3.402823e38 as proxy (FLT_MAX), meaning x2 is inf
    // Actually test isinf(x2): if x2 == x2 + 1.0 (inf + 1 == inf)
    // Simpler: check abs_x > sqrt(FLT_MAX) ≈ 1.8446726e19
    // fp32 can't represent 1.844e19 exactly, use 1.8e19f
    // Actually 1.844674e19f in fp32...
    // The largest fp32 < inf that when squared gives inf:
    // sqrt(3.4028235e38) = 1.8446743e19, so threshold = 1.8446743e19f
    v_if(abs_x > 1.8446726e19f) {
        result = large_result;
    }
    v_endif;

    return result;
}

// ---------------------------------------------------------------------------
// acosh(x) = log(x + sqrt(x^2 - 1))
//
// Numerically stable via log1p:
//   acosh(x) = log1p((x-1) + sqrt((x-1)*(x+1)))
//            = log1p((x-1) + sqrt(x-1)*sqrt(x+1))
//            = log1p(sqrt(x-1) * (sqrt(x-1) + sqrt(x+1)))
//
// Alternative stable form:
//   acosh(x) = log1p(x - 1 + sqrt((x-1)*(x+1)))
//   Let u = x - 1 (small when x → 1), v = x + 1
//   arg = u + sqrt(u * v) = u + sqrt(u) * sqrt(v)
//   acosh(x) = log1p(arg)
//
// For large x (x^2 overflows):
//   acosh(x) ≈ log(2x) = log(x) + log(2)
//   Threshold: x > sqrt(FLT_MAX) ≈ 1.844e19
//
// Domain: x >= 1. acosh(1) = 0, acosh(x<1) = NaN
// ---------------------------------------------------------------------------
template <bool APPROXIMATION_MODE>
sfpi_inline vFloat _calculate_acosh_body_(vFloat inp) {
    // u = x - 1, v = x + 1
    vFloat u = inp - vConst1;  // x - 1, small when x near 1
    vFloat v = inp + vConst1;  // x + 1

    // sqrt(u * v) = sqrt((x-1)(x+1)) = sqrt(x^2 - 1)
    // But u*v might lose precision when u is tiny and v is ~2.
    // We compute sqrt_uv directly: sqrt(u) * sqrt(v) avoids u*v underflow
    // when u is tiny (u ≥ 0 since x ≥ 1).
    //
    // For u very small: sqrt(u)*sqrt(v) ≈ sqrt(u)*sqrt(2) — no issue.
    // This is more stable than computing u*v first.
    vFloat sqrt_u = _calculate_sqrt_body_<APPROXIMATION_MODE>(u);
    vFloat sqrt_v = _calculate_sqrt_body_<APPROXIMATION_MODE>(v);
    vFloat sqrt_uv = sqrt_u * sqrt_v;

    // arg = u + sqrt(u*v) = (x-1) + sqrt((x-1)(x+1))
    vFloat arg = u + sqrt_uv;

    // acosh(x) = log1p(arg)
    vFloat result = _log1p_<APPROXIMATION_MODE>(arg);

    // Large x path: when x > sqrt(FLT_MAX), use log(x) + log(2)
    vFloat log_x = _calculate_log_body_no_init_(inp);
    vFloat large_result = log_x + 0.6931471805599453f;  // log(2)

    v_if(inp > 1.8446726e19f) {
        result = large_result;
    }
    v_endif;

    return result;
}

// ---------------------------------------------------------------------------
// Public kernel entry points
// ---------------------------------------------------------------------------

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_atanh() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[0];
        dst_reg[0] = _calculate_atanh_body_<APPROXIMATION_MODE>(val);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_asinh() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[0];
        dst_reg[0] = _calculate_asinh_body_<APPROXIMATION_MODE>(val);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_acosh() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[0];
        dst_reg[0] = _calculate_acosh_body_<APPROXIMATION_MODE>(val);
        dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
