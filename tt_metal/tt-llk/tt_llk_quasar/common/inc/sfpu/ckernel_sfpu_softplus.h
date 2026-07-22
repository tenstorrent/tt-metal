// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_converter.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_polyval.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

// Softplus via abs(x) symmetry (ported from Blackhole): with f(a) = ln(1+exp(-a)),
// softplus(t) = t + f(t) for t >= 0 and f(-t) for t < 0. is_fp32_dest_acc_en selects a degree-8 poly on
// [0,5] + exp Taylor tail (32-bit Dest) vs a bf16-accurate degree-6 poly with the tail dropped (16-bit).

constexpr float SOFTPLUS_POLY_BOUNDARY = 5.0f;

// FP32 residual polynomial: f(a) = ln(1+exp(-a)) on [0, 5], degree 8
constexpr float SOFTPLUS_POLY_C0 = 6.9310557842e-01f;
constexpr float SOFTPLUS_POLY_C1 = -4.9926245213e-01f;
constexpr float SOFTPLUS_POLY_C2 = 1.2186349183e-01f;
constexpr float SOFTPLUS_POLY_C3 = 5.6753782555e-03f;
constexpr float SOFTPLUS_POLY_C4 = -1.0528374463e-02f;
constexpr float SOFTPLUS_POLY_C5 = 2.7290175203e-03f;
constexpr float SOFTPLUS_POLY_C6 = -3.4358495031e-04f;
constexpr float SOFTPLUS_POLY_C7 = 2.1285692128e-05f;
constexpr float SOFTPLUS_POLY_C8 = -4.8245715334e-07f;

// BF16 residual polynomial: f(a) = ln(1+exp(-a)) on [0, 5], degree 6
// (ULP-weighted minimax fit; max error < 0.28 bf16 ULP over the domain)
constexpr float SOFTPLUS_BF16_POLY_C0 = 6.9423984729e-01f;
constexpr float SOFTPLUS_BF16_POLY_C1 = -5.0932420424e-01f;
constexpr float SOFTPLUS_BF16_POLY_C2 = 1.4279095486e-01f;
constexpr float SOFTPLUS_BF16_POLY_C3 = -1.3000584069e-02f;
constexpr float SOFTPLUS_BF16_POLY_C4 = -1.8627923291e-03f;
constexpr float SOFTPLUS_BF16_POLY_C5 = 5.0152968088e-04f;
constexpr float SOFTPLUS_BF16_POLY_C6 = -3.1273466851e-05f;

template <bool is_fp32_dest_acc_en>
sfpi_inline void _calculate_softplus_body_(const float beta, const float beta_reciprocal, const float threshold)
{
    sfpi::vFloat val = sfpi::dst_reg[0]; // load x from dest (SFPLOAD)
    sfpi::vFloat t   = beta * val;

    // Linear region (t >= threshold): softplus(x) = x; default for every lane so the single store covers it.
    sfpi::vFloat result = val;

    // `t < threshold` relies on vConstNeg1/LREG11 == -1.0 (re-established per launch by _init_sfpu_config_reg_).
    v_if (t < threshold)
    {
        sfpi::vFloat a = sfpi::abs(t);
        sfpi::vFloat residual;

        if constexpr (is_fp32_dest_acc_en)
        {
            residual = PolynomialEvaluator::eval(
                a,
                SOFTPLUS_POLY_C0,
                SOFTPLUS_POLY_C1,
                SOFTPLUS_POLY_C2,
                SOFTPLUS_POLY_C3,
                SOFTPLUS_POLY_C4,
                SOFTPLUS_POLY_C5,
                SOFTPLUS_POLY_C6,
                SOFTPLUS_POLY_C7,
                SOFTPLUS_POLY_C8);

            // Tail for a > 5: f(a) ~ exp(-a) via 3-term Taylor ln(1+e) = e*(1 + e*(-1/2 + e/3)).
            v_if (a > SOFTPLUS_POLY_BOUNDARY)
            {
                sfpi::vFloat e = _sfpu_exp_fp32_accurate_(-a);
                residual       = e * (1.0f + e * (-0.5f + e * 0.333333343f));
            }
            v_endif;
        }
        else
        {
            residual = PolynomialEvaluator::eval(
                a,
                SOFTPLUS_BF16_POLY_C0,
                SOFTPLUS_BF16_POLY_C1,
                SOFTPLUS_BF16_POLY_C2,
                SOFTPLUS_BF16_POLY_C3,
                SOFTPLUS_BF16_POLY_C4,
                SOFTPLUS_BF16_POLY_C5,
                SOFTPLUS_BF16_POLY_C6);

            // The degree-6 poly diverges past its [0, 5] fit domain, while the true residual <
            // exp(-5) = 0.0067 there; clamp to 0 to keep softplus(t>0) = t within bf16 rounding.
            v_if (a > SOFTPLUS_POLY_BOUNDARY)
            {
                residual = 0.0f;
            }
            v_endif;
        }

        // Reconstruct: t >= 0 -> max(0,t) + residual; t < 0 -> residual.
        sfpi::vFloat tp = sfpi::max(t, 0.0f);
        result          = beta_reciprocal * (tp + residual);

        // Round-to-nearest for a 16-bit Dest (SFPSTORE defaults to truncation).
        if constexpr (!is_fp32_dest_acc_en)
        {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
    }
    v_endif;

    sfpi::dst_reg[0] = result;
    sfpi::dst_reg++;
}

// Calculates SOFTPLUS over a full tile; param0/param1/param2 are beta, 1/beta, threshold (fp32 bit
// patterns). APPROXIMATION_MODE ignored (exact op).
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_softplus_(std::uint32_t param0, std::uint32_t param1, std::uint32_t param2)
{
    const float beta            = Converter::as_float(param0);
    const float beta_reciprocal = Converter::as_float(param1);
    const float threshold       = Converter::as_float(param2);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        _calculate_softplus_body_<is_fp32_dest_acc_en>(beta, beta_reciprocal, threshold);
    }
}

} // namespace ckernel::sfpu
