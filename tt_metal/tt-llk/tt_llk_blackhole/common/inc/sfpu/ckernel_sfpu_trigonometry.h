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
// Uses a split approach:
//   For |x| <= 0.5:
//     Uses the series via s = x/(2+x):
//       log1p(x) = 2*s*(1 + s^2/3 + s^4/5 + s^6/7)
//     where s = x/(x+2). This avoids catastrophic cancellation.
//
//   For |x| > 0.5:
//     log1p(x) = _calculate_log_body_no_init_(1 + x)
//     (no cancellation risk since 1+x is not close to 1 relative to its magnitude)
// ---------------------------------------------------------------------------
template <bool APPROXIMATION_MODE>
sfpi_inline vFloat _log1p_(vFloat x) {
    vFloat one_plus_x = vConst1 + x;

    // Series: s = x/(x+2), log1p(x) = 2*s*(1 + s^2*(1/3 + s^2*(1/5 + s^2/7)))
    vFloat two_plus_x = x + 2.0f;
    vFloat inv_two_plus_x = _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(two_plus_x);
    vFloat s = x * inv_two_plus_x;
    vFloat s2 = s * s;

    vFloat poly = s2 * (1.0f / 7.0f);
    poly = s2 * (poly + (1.0f / 5.0f));
    poly = s2 * (poly + (1.0f / 3.0f));
    poly = poly + 1.0f;
    vFloat series_result = 2.0f * s * poly;

    vFloat direct_result = _calculate_log_body_no_init_(one_plus_x);

    vFloat result = direct_result;
    v_if(x >= -0.5f && x < 0.5f) {
        result = series_result;
    }
    v_endif;

    return result;
}

// ---------------------------------------------------------------------------
// atanh(x) = 0.5 * log1p(2x / (1 - x))
//
// Avoids catastrophic cancellation for x → 0 and reciprocal blowup for x → ±1.
// ---------------------------------------------------------------------------
template <bool APPROXIMATION_MODE>
sfpi_inline vFloat _calculate_atanh_body_(vFloat inp) {
    vFloat one_minus_x = vConst1 - inp;
    vFloat inv_one_minus_x = _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(one_minus_x);
    vFloat arg = 2.0f * inp * inv_one_minus_x;
    vFloat result = 0.5f * _log1p_<APPROXIMATION_MODE>(arg);
    return result;
}

// ---------------------------------------------------------------------------
// asinh(x) = sign(x) * log1p(|x| + x^2 / (sqrt(x^2+1) + 1))
//
// Stable for small x (avoids log(1 + tiny)) and large x (separate path).
// ---------------------------------------------------------------------------
template <bool APPROXIMATION_MODE>
sfpi_inline vFloat _calculate_asinh_body_(vFloat inp) {
    vFloat abs_x = sfpi::abs(inp);
    vFloat x2 = inp * inp;

    vFloat sqrt_x2p1 = _calculate_sqrt_body_<APPROXIMATION_MODE>(x2 + vConst1);
    vFloat sqrt_p1_plus1 = sqrt_x2p1 + vConst1;
    vFloat inv_sqrt_p1_plus1 = _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(sqrt_p1_plus1);
    vFloat correction = x2 * inv_sqrt_p1_plus1;
    vFloat arg = abs_x + correction;

    vFloat log_val = _log1p_<APPROXIMATION_MODE>(arg);

    vFloat result = log_val;
    v_if(inp < 0.0f) {
        result = -log_val;
    }
    v_endif;

    // Large |x| path: asinh(x) ≈ sign(x) * (log(|x|) + log(2))
    vFloat log_abs_x = _calculate_log_body_no_init_(abs_x);
    vFloat large_result = log_abs_x + 0.6931471805599453f;
    v_if(inp < 0.0f) {
        large_result = -large_result;
    }
    v_endif;

    v_if(abs_x > 1.8446726e19f) {
        result = large_result;
    }
    v_endif;

    return result;
}

// ---------------------------------------------------------------------------
// acosh(x) = log1p((x-1) + sqrt(x-1)*sqrt(x+1))
//
// Stable for x → 1⁺ (avoids log(x + tiny) cancellation).
// Large x path avoids x^2 overflow.
// ---------------------------------------------------------------------------
template <bool APPROXIMATION_MODE>
sfpi_inline vFloat _calculate_acosh_body_(vFloat inp) {
    vFloat u = inp - vConst1;
    vFloat v = inp + vConst1;

    vFloat sqrt_u = _calculate_sqrt_body_<APPROXIMATION_MODE>(u);
    vFloat sqrt_v = _calculate_sqrt_body_<APPROXIMATION_MODE>(v);
    vFloat sqrt_uv = sqrt_u * sqrt_v;

    vFloat arg = u + sqrt_uv;
    vFloat result = _log1p_<APPROXIMATION_MODE>(arg);

    // Large x path
    vFloat log_x = _calculate_log_body_no_init_(inp);
    vFloat large_result = log_x + 0.6931471805599453f;

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
