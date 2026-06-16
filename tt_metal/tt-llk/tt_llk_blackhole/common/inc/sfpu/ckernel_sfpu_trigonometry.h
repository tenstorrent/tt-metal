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
// Strategy (Kahan / Higham):
//   u = 1 + x
//   if u == 1 (x tiny):  return x          (exact, no rounding error)
//   else:                 return x * log(u) / (u - 1)
//
// This preserves full precision for small x where log(1+x) ≈ x.
//
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _calculate_log1p_(sfpi::vFloat x) {
    sfpi::vFloat u = x + sfpi::vConst1;
    sfpi::vFloat u_m1 = u - sfpi::vConst1;

    sfpi::vFloat log_u = _calculate_log_body_no_init_(u);

    // ratio = x * log(u) / (u - 1)
    sfpi::vFloat ratio = x * log_u;
    sfpi::vFloat result = ratio * _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(u_m1);

    // When u - 1 rounds to 0, log1p(x) = x
    v_if(u_m1 == sfpi::vConst0) { result = x; }
    v_endif;

    return result;
}

//
// atanh(x) = 0.5 * log1p(2x / (1 - x))
//
// Avoids:
//  - catastrophic cancellation in log near x=0
//  - reciprocal blow-up near x=±1 (log produces ±inf naturally)
//
template <bool APPROXIMATION_MODE>
sfpi_inline void _calculate_atanh_body_(sfpi::vFloat inp) {
    sfpi::vFloat one_minus_x = sfpi::vConst1 - inp;
    sfpi::vFloat two_x = inp + inp;
    sfpi::vFloat arg = two_x * _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(one_minus_x);

    sfpi::vFloat l = _calculate_log1p_<APPROXIMATION_MODE>(arg);
    sfpi::dst_reg[0] = l * 0.5f;
}

//
// asinh(x) = sign(x) * log1p(|x| + x^2 / (1 + sqrt(1 + x^2)))
//
// For large |x| > LARGE_THRESHOLD: sign(x) * (log(2) + log(|x|))
//
template <bool APPROXIMATION_MODE>
sfpi_inline void _calculate_asinh_body_(sfpi::vFloat inp) {
    const float LARGE_THRESHOLD = 1.3e19f;
    const float LOG2_F = 0.6931471805599453f;

    sfpi::vFloat abs_x = sfpi::abs(inp);

    // x^2
    sfpi::vFloat x2 = inp * inp;
    sfpi::vFloat x2p1 = x2 + sfpi::vConst1;
    sfpi::vFloat sq = _calculate_sqrt_body_<APPROXIMATION_MODE>(x2p1);

    // arg = |x| + x^2 / (1 + sqrt(1 + x^2))
    sfpi::vFloat one_plus_sq = sq + sfpi::vConst1;
    sfpi::vFloat x2_div = x2 * _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(one_plus_sq);
    sfpi::vFloat arg = abs_x + x2_div;

    sfpi::vFloat log_val = _calculate_log1p_<APPROXIMATION_MODE>(arg);

    // Large x path: log(2|x|) = log(2) + log(|x|)
    sfpi::vFloat log_large = _calculate_log_body_no_init_(abs_x);
    log_large = log_large + LOG2_F;

    sfpi::vFloat result = log_val;
    v_if(abs_x > LARGE_THRESHOLD) { result = log_large; }
    v_endif;

    // Apply sign of input
    // sign(inp) implemented as: inp >= 0 ? 1 : -1
    sfpi::vFloat sign_x = sfpi::vConst1;
    v_if(inp < sfpi::vConst0) { sign_x = sfpi::vFloat(-1.0f); }
    v_endif;

    sfpi::vFloat signed_result = result * sign_x;

    // x == 0 -> 0
    v_if(inp == sfpi::vConst0) { signed_result = sfpi::vConst0; }
    v_endif;

    sfpi::dst_reg[0] = signed_result;
}

//
// acosh(x) = log1p((x-1) + sqrt((x-1)*(x+1))),  domain x >= 1
//
// Avoids:
//  - x^2 overflow for large x  (uses (x-1)*(x+1) instead)
//  - cancellation near x=1     (x-1 computed exactly, log1p used)
//  - large x: log(2) + log(x)
//
template <bool APPROXIMATION_MODE>
sfpi_inline void _calculate_acosh_body_(sfpi::vFloat inp) {
    const float LARGE_THRESHOLD = 1.3e19f;
    const float LOG2_F = 0.6931471805599453f;

    sfpi::vFloat t = inp - sfpi::vConst1;                       // x - 1
    sfpi::vFloat t_plus_2 = t + sfpi::vConst1 + sfpi::vConst1; // x + 1
    sfpi::vFloat inner = t * t_plus_2;                          // (x-1)(x+1) = x^2-1, no overflow

    sfpi::vFloat sq = _calculate_sqrt_body_<APPROXIMATION_MODE>(inner);
    sfpi::vFloat arg = t + sq;

    sfpi::vFloat log_val = _calculate_log1p_<APPROXIMATION_MODE>(arg);

    sfpi::vFloat log_large = _calculate_log_body_no_init_(inp);
    log_large = log_large + LOG2_F;

    sfpi::vFloat result = log_val;
    v_if(inp > LARGE_THRESHOLD) { result = log_large; }
    v_endif;

    v_if(inp == sfpi::vConst1) { result = sfpi::vConst0; }
    v_endif;

    sfpi::dst_reg[0] = result;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_atanh() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = dst_reg[0];
        _calculate_atanh_body_<APPROXIMATION_MODE>(v);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_asinh() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = dst_reg[0];
        _calculate_asinh_body_<APPROXIMATION_MODE>(v);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_acosh() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = dst_reg[0];
        _calculate_acosh_body_<APPROXIMATION_MODE>(v);
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
