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
// For |x| < 0.5:  use u = x/(2+x), then log(1+x) = 2u*(1 + u^2/3 + u^4/5 + ...)
// For |x| >= 0.5: delegate to _calculate_log_body_(1+x)
//
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
sfpi_inline sfpi::vFloat _calculate_log1p_body_(sfpi::vFloat x) {
    sfpi::vFloat one_plus_x = x + sfpi::vConst1;

    // Path A: direct log
    sfpi::vFloat log_direct = _calculate_log_body_<APPROXIMATION_MODE>(one_plus_x);

    // Path B: polynomial via u = x/(2+x)
    sfpi::vFloat two_plus_x = x + sfpi::vFloat(2.0f);
    sfpi::vFloat u = x * _sfpu_recip_(two_plus_x);
    sfpi::vFloat u2 = u * u;

    sfpi::vFloat poly = sfpi::vFloat(1.0f / 15.0f);
    poly = sfpi::vFloat(1.0f / 13.0f) + u2 * poly;
    poly = sfpi::vFloat(1.0f / 11.0f) + u2 * poly;
    poly = sfpi::vFloat(1.0f / 9.0f) + u2 * poly;
    poly = sfpi::vFloat(1.0f / 7.0f) + u2 * poly;
    poly = sfpi::vFloat(1.0f / 5.0f) + u2 * poly;
    poly = sfpi::vFloat(1.0f / 3.0f) + u2 * poly;
    poly = sfpi::vFloat(1.0f) + u2 * poly;
    sfpi::vFloat log_poly = sfpi::vFloat(2.0f) * u * poly;

    sfpi::vFloat abs_x = sfpi::abs(x);
    sfpi::vFloat result = log_direct;
    v_if(abs_x < sfpi::vFloat(0.5f)) { result = log_poly; }
    v_endif;

    return result;
}

//
// atanh(x) = 0.5 * log1p(2x / (1 - x))
//
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
sfpi_inline sfpi::vFloat _calculate_atanh_body_(sfpi::vFloat x) {
    sfpi::vFloat one_minus_x = sfpi::vConst1 - x;
    sfpi::vFloat two_x = x + x;
    sfpi::vFloat u = two_x * _sfpu_recip_(one_minus_x);
    sfpi::vFloat log1p_u = _calculate_log1p_body_<APPROXIMATION_MODE>(u);
    return sfpi::vFloat(0.5f) * log1p_u;
}

//
// asinh(x) = log1p(x + x^2 / (1 + sqrt(1 + x^2)))
//
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
sfpi_inline sfpi::vFloat _calculate_asinh_body_(sfpi::vFloat x) {
    sfpi::vFloat x2 = x * x;
    sfpi::vFloat one_plus_x2 = x2 + sfpi::vConst1;
    sfpi::vFloat t = _calculate_sqrt_body_<APPROXIMATION_MODE>(one_plus_x2);
    sfpi::vFloat one_plus_t = t + sfpi::vConst1;
    sfpi::vFloat u = x + x2 * _sfpu_recip_(one_plus_t);
    return _calculate_log1p_body_<APPROXIMATION_MODE>(u);
}

//
// acosh(x) = log1p((x-1) + sqrt((x-1)*(x+1))),  x >= 1
//
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
sfpi_inline sfpi::vFloat _calculate_acosh_body_(sfpi::vFloat x) {
    sfpi::vFloat d = x - sfpi::vConst1;
    sfpi::vFloat d_plus_2 = d + sfpi::vFloat(2.0f);
    sfpi::vFloat radicand = d * d_plus_2;

    sfpi::vFloat zero = sfpi::vFloat(0.0f);
    v_if(radicand < zero) { radicand = zero; }
    v_endif;

    sfpi::vFloat sq = _calculate_sqrt_body_<APPROXIMATION_MODE>(radicand);
    sfpi::vFloat u = d + sq;
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

}  // namespace sfpu
}  // namespace ckernel
