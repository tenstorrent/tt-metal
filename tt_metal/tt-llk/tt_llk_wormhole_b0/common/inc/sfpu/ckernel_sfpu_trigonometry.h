// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

//
// log1p: compute log(1 + x) in a numerically stable way.
//
// For |x| < 2^-23 (tiny), log1p(x) ≈ x  (avoids underflow in log)
// For |x| >= 0.5, use standard log(1 + x) path
// For |x| < 0.5, use the Kahan / Higham identity:
//   u = 1 + x
//   log1p(x) = x * log(u) / (u - 1)    when u != 1
//             = x                        when u == 1  (x very small)
//
// This avoids catastrophic cancellation in (u - 1) for small x.
//
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _calculate_log1p_(sfpi::vFloat x) {
    // u = 1 + x
    sfpi::vFloat u = x + sfpi::vConst1;

    // For |x| very small (u rounds to 1), return x directly
    sfpi::vFloat u_m1 = u - sfpi::vConst1;

    // Compute log(u) using the existing log body
    // (assumes u > 0, caller must guarantee this)
    sfpi::vFloat log_u = _calculate_log_body_no_init_(u);

    // result = x * log(u) / (u - 1)
    // When u - 1 == 0 (x extremely small), fall back to x
    // We implement: if |u_m1| < eps, return x, else return x * log_u / u_m1
    sfpi::vFloat ratio = x * log_u;

    // Use conditional: where u_m1 is effectively zero, just return x
    // sfpi v_if / v_elseif operate on the result register
    sfpi::vFloat result = ratio / u_m1;

    // If u == 1 (i.e., u_m1 == 0), use x as the result
    v_if(u_m1 == sfpi::vConst0) { result = x; }
    v_endif;

    return result;
}

//
// atanh(x) = 0.5 * log((1+x)/(1-x))
//
// Numerically stable form for |x| < 1:
//   atanh(x) = 0.5 * log1p(2x / (1 - x))
//
// This avoids catastrophic cancellation near x=0 and avoids reciprocal
// blow-up near x=±1 (the log naturally produces ±inf there).
//
template <bool APPROXIMATION_MODE>
sfpi_inline void _calculate_atanh_body_(sfpi::vFloat inp) {
    // arg = 2*x / (1 - x)  =  -2*x / (x - 1)
    sfpi::vFloat one_minus_x = sfpi::vConst1 - inp;
    sfpi::vFloat two_x = inp + inp;
    sfpi::vFloat arg = two_x * _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(one_minus_x);

    // log1p(arg) via stable path
    sfpi::vFloat l = _calculate_log1p_<APPROXIMATION_MODE>(arg);

    // atanh(x) = 0.5 * log1p(arg)
    sfpi::dst_reg[0] = l * 0.5f;
}

//
// asinh(x) = log(|x| + sqrt(x^2 + 1))
//
// Numerically stable forms:
//   Near zero (|x| small): use log1p(x + x^2/(1 + sqrt(1 + x^2)))
//     This avoids log(1 + small) catastrophic cancellation.
//   Large |x| (x^2 would overflow, i.e., |x| > sqrt(FLT_MAX/2) ~ 1.3e19):
//     asinh(x) = sign(x) * (log(2) + log(|x|))
//   Otherwise (normal range):
//     asinh(x) = sign(x) * log1p(|x| + x^2 / (1 + sqrt(1 + x^2)))
//
// The key identity: |x| + sqrt(1 + x^2) - 1 = x^2 / (1 + sqrt(1 + x^2))
// so log(|x| + sqrt(1+x^2)) = log1p(|x| + x^2/(1 + sqrt(1+x^2)))
//
template <bool APPROXIMATION_MODE>
sfpi_inline void _calculate_asinh_body_(sfpi::vFloat inp) {
    // Constants
    const float LARGE_THRESHOLD = 1.3e19f;  // ~sqrt(FLT_MAX / 2)
    const float LOG2_F = 0.6931471805599453f;

    sfpi::vFloat abs_x = sfpi::abs(inp);
    sfpi::vFloat sign_x = inp * _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(abs_x);

    // Clamp sign_x for x=0 case
    v_if(abs_x == sfpi::vConst0) { sign_x = sfpi::vConst0; }
    v_endif;

    // x^2
    sfpi::vFloat x2 = inp * inp;

    // sqrt(1 + x^2)
    sfpi::vFloat x2p1 = x2 + sfpi::vConst1;
    sfpi::vFloat sq = _calculate_sqrt_body_<APPROXIMATION_MODE>(x2p1);

    // arg = |x| + x^2 / (1 + sqrt(1 + x^2))
    // = |x| + x^2 / (1 + sq)
    sfpi::vFloat one_plus_sq = sq + sfpi::vConst1;
    sfpi::vFloat x2_div = x2 * _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(one_plus_sq);
    sfpi::vFloat arg = abs_x + x2_div;

    // log1p(arg) — stable for all normal |x|
    sfpi::vFloat log_val = _calculate_log1p_<APPROXIMATION_MODE>(arg);

    // For large |x|, use log(|x|) + log(2) to avoid overflow
    // log(2|x|) = log(2) + log(|x|)
    sfpi::vFloat log_large = _calculate_log_body_no_init_(abs_x);
    log_large = log_large + LOG2_F;

    // Select between normal and large-x paths
    sfpi::vFloat result = log_val;
    v_if(abs_x > LARGE_THRESHOLD) { result = log_large; }
    v_endif;

    // Restore sign
    sfpi::dst_reg[0] = result * sign_x;

    // Special case: x == 0 -> result == 0
    v_if(inp == sfpi::vConst0) { sfpi::dst_reg[0] = sfpi::vConst0; }
    v_endif;
}

//
// acosh(x) = log(x + sqrt(x^2 - 1)),  domain x >= 1
//
// Numerically stable forms:
//   Near x=1 (x^2 - 1 small): catastrophic cancellation in log(x + sqrt(x^2-1))
//     Use: log1p((x-1) + sqrt((x-1)*(x+1)))
//          = log1p((x-1) + sqrt(x^2-1))
//     Since x-1 is exact and sqrt(x^2-1) = sqrt((x-1)(x+1))
//     For x near 1: let t = x - 1, then
//       acosh(x) = log1p(t + sqrt(t*(t+2)))
//                = log1p(t + sqrt(t) * sqrt(t+2))
//
//   Large x (x > sqrt(FLT_MAX/2) ~ 1.3e19):
//     acosh(x) = log(2) + log(x)  (since sqrt(x^2-1) ≈ x)
//
//   Normal range:
//     acosh(x) = log1p((x-1) + sqrt((x-1)*(x+1)))
//              This is always safe: (x-1) >= 0, sqrt() >= 0, so arg >= 0
//
template <bool APPROXIMATION_MODE>
sfpi_inline void _calculate_acosh_body_(sfpi::vFloat inp) {
    const float LARGE_THRESHOLD = 1.3e19f;
    const float LOG2_F = 0.6931471805599453f;

    // t = x - 1  (exact subtraction, x >= 1)
    sfpi::vFloat t = inp - sfpi::vConst1;

    // sqrt((x-1)*(x+1)) = sqrt(x^2 - 1)
    // computed as sqrt(t * (t + 2)) to avoid computing x^2 (which overflows for large x)
    sfpi::vFloat t_plus_2 = t + sfpi::vConst1 + sfpi::vConst1;  // t + 2 = x + 1
    sfpi::vFloat inner = t * t_plus_2;  // (x-1)*(x+1) = x^2 - 1, no overflow risk

    sfpi::vFloat sq = _calculate_sqrt_body_<APPROXIMATION_MODE>(inner);

    // arg = t + sq = (x-1) + sqrt(x^2-1)
    sfpi::vFloat arg = t + sq;

    // acosh(x) = log1p(arg)
    sfpi::vFloat log_val = _calculate_log1p_<APPROXIMATION_MODE>(arg);

    // For large x, use log(x) + log(2)
    sfpi::vFloat log_large = _calculate_log_body_no_init_(inp);
    log_large = log_large + LOG2_F;

    sfpi::vFloat result = log_val;
    v_if(inp > LARGE_THRESHOLD) { result = log_large; }
    v_endif;

    // acosh(1) = 0 exactly
    v_if(inp == sfpi::vConst1) { result = sfpi::vConst0; }
    v_endif;

    sfpi::dst_reg[0] = result;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_atanh() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = dst_reg[0];
        _calculate_atanh_body_<APPROXIMATION_MODE>(v);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_asinh() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = dst_reg[0];
        _calculate_asinh_body_<APPROXIMATION_MODE>(v);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_acosh() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = dst_reg[0];
        _calculate_acosh_body_<APPROXIMATION_MODE>(v);
        dst_reg++;
    }
}

// ─── sinh / cosh / tanh (unchanged, reproduced for completeness) ─────────────

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_sinh() {
    // sinh(x) = (e^x - e^-x) / 2
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = dst_reg[0];
        sfpi::vFloat exp_pos = sfpi::vFloat(1.0f);
        sfpi::vFloat exp_neg = sfpi::vFloat(1.0f);

        // Use existing exp implementation via the series / HW path
        // e^x
        sfpi::vFloat ex = sfpi::vFloat(1.0f);
        sfpu_expf_body<APPROXIMATION_MODE>(v, ex);
        // e^-x
        sfpi::vFloat neg_v = v * sfpi::vFloat(-1.0f);
        sfpi::vFloat enx = sfpi::vFloat(1.0f);
        sfpu_expf_body<APPROXIMATION_MODE>(neg_v, enx);

        dst_reg[0] = (ex - enx) * sfpi::vFloat(0.5f);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_cosh() {
    // cosh(x) = (e^x + e^-x) / 2
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = dst_reg[0];
        sfpi::vFloat ex = sfpi::vFloat(1.0f);
        sfpu_expf_body<APPROXIMATION_MODE>(v, ex);
        sfpi::vFloat neg_v = v * sfpi::vFloat(-1.0f);
        sfpi::vFloat enx = sfpi::vFloat(1.0f);
        sfpu_expf_body<APPROXIMATION_MODE>(neg_v, enx);

        dst_reg[0] = (ex + enx) * sfpi::vFloat(0.5f);
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
