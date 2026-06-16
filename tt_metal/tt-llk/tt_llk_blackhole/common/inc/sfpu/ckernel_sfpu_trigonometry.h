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
// For |x| < 0.5:  use Horner-form Taylor series (6 terms, fp32-accurate):
//   log1p(x) ≈ x*(1 - x*(1/2 - x*(1/3 - x*(1/4 - x*(1/5 - x/6)))))
//
// For |x| >= 0.5: 1+x is exact (no cancellation), delegate to standard log.
//
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _calculate_log1p_body_(sfpi::vFloat x) {
    sfpi::vFloat one = sfpi::vConst1;

    // Horner evaluation of log1p polynomial for |x| < 0.5
    sfpi::vFloat p = sfpi::vFloat(1.0f / 6.0f);
    p = sfpi::vFloat(1.0f / 5.0f) - x * p;
    p = sfpi::vFloat(1.0f / 4.0f) - x * p;
    p = sfpi::vFloat(1.0f / 3.0f) - x * p;
    p = sfpi::vFloat(0.5f)        - x * p;
    p = one                       - x * p;
    sfpi::vFloat poly_result = x * p;

    // For |x| >= 0.5: compute log(1+x) via standard log (no cancellation)
    sfpi::vFloat arg        = x + one;
    sfpi::vFloat log_result = _calculate_log_body_no_init_(arg);

    // Branch on |x| < 0.5
    sfpi::vFloat abs_x = sfpi::abs(x);
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
// Avoids cancellation near x=0 and reciprocal blow-up near x=±1.
//
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _calculate_atanh_body_(sfpi::vFloat inp) {
    sfpi::vFloat one_minus_x = sfpi::vConst1 - inp;

    sfpi::vFloat recip_den;
    if constexpr (APPROXIMATION_MODE) {
        recip_den = _sfpu_reciprocal_<0>(one_minus_x);
    } else {
        recip_den = _sfpu_reciprocal_<2>(one_minus_x);
    }

    sfpi::vFloat t        = (sfpi::vFloat(2.0f) * inp) * recip_den;
    sfpi::vFloat log1p_t  = _calculate_log1p_body_<APPROXIMATION_MODE>(t);

    return sfpi::vFloat(0.5f) * log1p_t;
}

//
// asinh(x) = log(|x| + sqrt(x^2 + 1)) * sign(x)
//
// Numerically stable via Kahan's formula:
//   Small |x| (<=1): log1p(|x| + x^2/(sqrt(1+x^2)+1))
//   Large |x| (>1, x^2 safe): log(2) + log(|x|) + log1p(1/(2x^2))
//   Very large |x| (>1.3e19, x^2 overflows): log(2) + log(|x|)
//
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _calculate_asinh_body_(sfpi::vFloat inp) {
    sfpi::vFloat abs_inp = sfpi::abs(inp);
    sfpi::vFloat x2      = abs_inp * abs_inp;

    sfpi::vFloat one_plus_x2 = x2 + sfpi::vConst1;
    sfpi::vFloat sq          = _calculate_sqrt_body_<APPROXIMATION_MODE>(one_plus_x2);

    // Small branch: Kahan formula
    sfpi::vFloat denom_small = sq + sfpi::vConst1;
    sfpi::vFloat recip_small;
    if constexpr (APPROXIMATION_MODE) {
        recip_small = _sfpu_reciprocal_<0>(denom_small);
    } else {
        recip_small = _sfpu_reciprocal_<2>(denom_small);
    }
    sfpi::vFloat kahan_arg = abs_inp + x2 * recip_small;
    sfpi::vFloat res_small = _calculate_log1p_body_<APPROXIMATION_MODE>(kahan_arg);

    // Large branch: log(2) + log(|x|) + log1p(1/(2x^2))
    sfpi::vFloat log2     = sfpi::vFloat(0.6931471805599453f);
    sfpi::vFloat log_absx = _calculate_log_body_no_init_(abs_inp);

    sfpi::vFloat inv_2x2;
    if constexpr (APPROXIMATION_MODE) {
        inv_2x2 = _sfpu_reciprocal_<0>(sfpi::vFloat(2.0f) * x2);
    } else {
        inv_2x2 = _sfpu_reciprocal_<2>(sfpi::vFloat(2.0f) * x2);
    }
    sfpi::vFloat log1p_inv_2x2 = _calculate_log1p_body_<APPROXIMATION_MODE>(inv_2x2);
    sfpi::vFloat res_large      = log2 + log_absx + log1p_inv_2x2;

    // Very large branch: log(2) + log(|x|)
    sfpi::vFloat res_vlarge = log2 + log_absx;

    // Select
    sfpi::vFloat result = res_small;
    v_if(abs_inp > sfpi::vFloat(1.0f)) {
        result = res_large;
    }
    v_endif;
    v_if(abs_inp > sfpi::vFloat(1.3e19f)) {
        result = res_vlarge;
    }
    v_endif;

    // Restore sign
    v_if(inp < sfpi::vFloat(0.0f)) {
        result = -result;
    }
    v_endif;

    return result;
}

//
// acosh(x) = log(x + sqrt(x^2 - 1))
//
// Numerically stable:
//   Near x=1 (x in [1,2]): log1p((x-1) + sqrt(x^2-1))
//   Normal (x > 2):         log(x + sqrt(x^2-1))
//   Very large (x > 1.3e19, x^2 overflows): log(2) + log(x)
//
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _calculate_acosh_body_(sfpi::vFloat inp) {
    sfpi::vFloat x2_minus_1 = inp * inp - sfpi::vConst1;

    // Clamp x^2-1 to 0 (handles x=1+eps rounding)
    sfpi::vFloat x2m1_clamped = x2_minus_1;
    v_if(x2_minus_1 < sfpi::vFloat(0.0f)) {
        x2m1_clamped = sfpi::vFloat(0.0f);
    }
    v_endif;
    sfpi::vFloat sq = _calculate_sqrt_body_<APPROXIMATION_MODE>(x2m1_clamped);

    // Near-1 branch: log1p((x-1) + sqrt(x^2-1))
    sfpi::vFloat xm1       = inp - sfpi::vConst1;
    sfpi::vFloat arg_near1 = xm1 + sq;
    sfpi::vFloat res_near1 = _calculate_log1p_body_<APPROXIMATION_MODE>(arg_near1);

    // Normal branch: log(x + sqrt(x^2-1))
    sfpi::vFloat arg_normal = inp + sq;
    sfpi::vFloat res_normal = _calculate_log_body_no_init_(arg_normal);

    // Very large branch: log(2) + log(x)
    sfpi::vFloat log2      = sfpi::vFloat(0.6931471805599453f);
    sfpi::vFloat log_inp   = _calculate_log_body_no_init_(inp);
    sfpi::vFloat res_vlarge = log2 + log_inp;

    // Select
    sfpi::vFloat result = res_near1;
    v_if(inp > sfpi::vFloat(2.0f)) {
        result = res_normal;
    }
    v_endif;
    v_if(inp > sfpi::vFloat(1.3e19f)) {
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
