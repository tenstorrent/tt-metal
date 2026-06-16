// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_converter.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_log.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_sqrt.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

//
// log1p(x) = log(1 + x), computed accurately for small |x|.
//
// Strategy:
//   - For |x| < 0.5: use the identity
//       log(1+x) = x - x^2/2 + x^3/3 - ...   (too slow for hardware)
//     Instead we use: let u = x/(2+x), then
//       log(1+x) = 2*u * (1 + u^2/3 + u^4/5 + u^6/7 + ...)
//     This is the classic Padé/Taylor approach used in glibc/fdlibm.
//   - For |x| >= 0.5: delegate directly to the existing log kernel,
//       log1p(x) = log(1 + x)
//
// We implement a hardware-friendly version that avoids branches by
// blending the two paths with a select.
//
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
sfpi_inline sfpi::vFloat _calculate_log1p_body_(sfpi::vFloat x) {
    // Compute 1 + x
    sfpi::vFloat one_plus_x = x + sfpi::vConst1;

    // Path A: direct log(1+x) via existing log kernel (accurate for |x| >= 0.5)
    sfpi::vFloat log_direct = _calculate_log_body_<APPROXIMATION_MODE>(one_plus_x);

    // Path B: polynomial via u = x/(2+x) for small |x|
    // u = x / (2 + x)
    sfpi::vFloat two_plus_x = x + sfpi::vFloat(2.0f);
    sfpi::vFloat u = x * _sfpu_recip_(two_plus_x);
    sfpi::vFloat u2 = u * u;

    // Horner: 1 + u2*(1/3 + u2*(1/5 + u2*(1/7 + u2*(1/9 + u2*(1/11 + u2*(1/13 + u2/15))))))
    sfpi::vFloat poly = sfpi::vFloat(1.0f / 15.0f);
    poly = sfpi::vFloat(1.0f / 13.0f) + u2 * poly;
    poly = sfpi::vFloat(1.0f / 11.0f) + u2 * poly;
    poly = sfpi::vFloat(1.0f / 9.0f) + u2 * poly;
    poly = sfpi::vFloat(1.0f / 7.0f) + u2 * poly;
    poly = sfpi::vFloat(1.0f / 5.0f) + u2 * poly;
    poly = sfpi::vFloat(1.0f / 3.0f) + u2 * poly;
    poly = sfpi::vFloat(1.0f) + u2 * poly;
    sfpi::vFloat log_poly = sfpi::vFloat(2.0f) * u * poly;

    // Select: use polynomial when |x| < 0.5, else use direct log
    // |x| < 0.5  <=>  x < 0.5 && x > -0.5
    // We use a blend: abs(x) < 0.5
    sfpi::vFloat abs_x = sfpi::abs(x);
    // v_if abs_x < 0.5 → use poly, else use direct
    sfpi::vFloat result = log_direct;
    v_if(abs_x < sfpi::vFloat(0.5f)) { result = log_poly; }
    v_endif;

    return result;
}

//
// atanh(x) = 0.5 * log((1+x)/(1-x))
//
// Numerically stable form (Kahan 1987):
//   atanh(x) = 0.5 * log1p(2x / (1 - x))
//
// This avoids catastrophic cancellation for small x because
// 2x/(1-x) is small when x is small, and log1p handles that accurately.
//
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
sfpi_inline sfpi::vFloat _calculate_atanh_body_(sfpi::vFloat x) {
    // u = 2x / (1 - x)
    sfpi::vFloat one_minus_x = sfpi::vConst1 - x;
    sfpi::vFloat two_x = x + x;
    sfpi::vFloat u = two_x * _sfpu_recip_(one_minus_x);

    // result = 0.5 * log1p(u)
    sfpi::vFloat log1p_u = _calculate_log1p_body_<APPROXIMATION_MODE>(u);
    return sfpi::vFloat(0.5f) * log1p_u;
}

//
// asinh(x) = log(x + sqrt(x^2 + 1))
//
// Numerically stable form:
//   asinh(x) = log1p(x + x^2 / (1 + sqrt(1 + x^2)))
//
// For small x: x^2/(1+sqrt(1+x^2)) ≈ x^2/2, so argument ≈ x + x^2/2 ≈ x (small),
// and log1p handles that accurately.
// For large x: the formula also remains well-conditioned.
//
// Reference: Kahan, "Branch Cuts for Complex Elementary Functions" (1987)
//
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
sfpi_inline sfpi::vFloat _calculate_asinh_body_(sfpi::vFloat x) {
    // t = sqrt(1 + x^2)
    sfpi::vFloat x2 = x * x;
    sfpi::vFloat one_plus_x2 = x2 + sfpi::vConst1;
    sfpi::vFloat t = _calculate_sqrt_body_<APPROXIMATION_MODE>(one_plus_x2);

    // u = x + x^2 / (1 + t)
    sfpi::vFloat one_plus_t = t + sfpi::vConst1;
    sfpi::vFloat u = x + x2 * _sfpu_recip_(one_plus_t);

    // result = log1p(u)
    return _calculate_log1p_body_<APPROXIMATION_MODE>(u);
}

//
// acosh(x) = log(x + sqrt(x^2 - 1)),  x >= 1
//
// Numerically stable form:
//   acosh(x) = log1p((x - 1) + sqrt((x-1)*(x+1)))
//            = log1p((x-1) + sqrt(x^2 - 1))
//
// Near x=1: x-1 ≈ 0, sqrt(x^2-1) ≈ 0, so the argument to log1p is small
// and log1p handles it accurately.
//
// Alternative stable form (used here):
//   Let d = x - 1
//   acosh(x) = log1p(d + sqrt(d*(d + 2)))
//            = log1p(d + sqrt(d*d + 2*d))
//
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
sfpi_inline sfpi::vFloat _calculate_acosh_body_(sfpi::vFloat x) {
    // d = x - 1  (small near x=1)
    sfpi::vFloat d = x - sfpi::vConst1;

    // sqrt((x-1)*(x+1)) = sqrt(d*(d+2)) = sqrt(d^2 + 2d)
    sfpi::vFloat d_plus_2 = d + sfpi::vFloat(2.0f);
    sfpi::vFloat radicand = d * d_plus_2;  // = (x-1)*(x+1) = x^2-1

    // Clamp radicand to >= 0 to avoid NaN for x slightly below 1 due to rounding
    sfpi::vFloat zero = sfpi::vFloat(0.0f);
    v_if(radicand < zero) { radicand = zero; }
    v_endif;

    sfpi::vFloat sq = _calculate_sqrt_body_<APPROXIMATION_MODE>(radicand);

    // u = d + sq  (both small near x=1)
    sfpi::vFloat u = d + sq;

    // result = log1p(u)
    return _calculate_log1p_body_<APPROXIMATION_MODE>(u);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_atanh(const uint dst_offset) {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = dst_reg[dst_offset + d];
        dst_reg[dst_offset + d] = _calculate_atanh_body_<APPROXIMATION_MODE>(v);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_asinh(const uint dst_offset) {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = dst_reg[dst_offset + d];
        dst_reg[dst_offset + d] = _calculate_asinh_body_<APPROXIMATION_MODE>(v);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_acosh(const uint dst_offset) {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = dst_reg[dst_offset + d];
        dst_reg[dst_offset + d] = _calculate_acosh_body_<APPROXIMATION_MODE>(v);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_trig(const uint dst_offset) {
    // placeholder for direct sin/cos via existing implementations
}

}  // namespace sfpu
}  // namespace ckernel
