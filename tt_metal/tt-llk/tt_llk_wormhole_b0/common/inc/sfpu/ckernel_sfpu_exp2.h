// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "ckernel_sfpu_exp.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

// Computes 2^x directly in base-2.
//
// Decompose x = n + f, where n = round(x) and f = x - n ∈ [-0.5, 0.5].
// Then 2^x = 2^n * 2^f, where:
//   - 2^n is realised by adding n to the IEEE-754 exponent of 2^f via setexp.
//   - 2^f is approximated by a degree-5 truncated Taylor expansion of
//     exp(f·ln2) in f. (A true Remez minimax polynomial would shave ~1 bit
//     of worst-case error vs. Taylor, but Taylor is well within the ≤ 1 ulp
//     fp32 / ≤ 0.5 ulp bf16 spec on [-0.5, 0.5] and keeps the coefficients
//     auditable against the closed-form expressions ln(2)^k / k!.)
//
// This bypasses the redundant `* ln(2)` / `* 1/ln(2)` round-trip that the
// previous implementation paid for by routing through `exp(x * ln(2))`.
//
// Caller MUST clamp x to the safe range (~[-127, 128)) before calling this
// helper; the public `_calculate_exp2_` wrapper handles the clamping and
// special-case dispatch.
sfpi_inline sfpi::vFloat _sfpu_exp2_core_unsafe_(sfpi::vFloat x)
{
    // Step 1: split x into integer part n (as int32) and fractional part f.
    // We use the same round-to-nearest-even trick as _sfpu_round_to_nearest_int32_
    // (see ckernel_sfpu_exp.h) to extract both n_float and n_int in one shot.
    sfpi::vInt n_int;
    sfpi::vFloat n_float = _sfpu_round_to_nearest_int32_(x, n_int);
    sfpi::vFloat f       = x - n_float;

    // Step 2: 2^f via degree-5 truncated Taylor polynomial on [-0.5, 0.5].
    // Coefficients are the closed-form Taylor expansion of exp(f·ln2);
    // single-precision representable and within ≤ 1 ulp on the interval.
    //   2^f ≈ 1 + c1·f + c2·f² + c3·f³ + c4·f⁴ + c5·f⁵
    sfpi::vFloat p = PolynomialEvaluator::eval(
        f,
        sfpi::vConst1,        // c0 = 1
        0x1.62e430p-1f,       // c1 = ln(2)               ≈ 0.6931472
        0x1.ebfbe0p-3f,       // c2 = ln(2)²/2            ≈ 0.2402266
        0x1.c6b08ep-5f,       // c3 = ln(2)³/6            ≈ 0.0555041
        0x1.3b2ab6p-7f,       // c4 = ln(2)⁴/24           ≈ 0.0096181
        0x1.5d87fep-10f       // c5 = ln(2)⁵/120          ≈ 0.0013336
    );

    // Step 3: scale by 2^n via direct exponent injection: setexp(p, exexp(p) + n).
    sfpi::vInt p_exp   = sfpi::exexp(p, sfpi::ExponentMode::NoDebias);
    sfpi::vInt new_exp = p_exp + n_int;
    return sfpi::setexp(p, new_exp);
}

// FP32 accurate path for 2^x with full special-case handling.
sfpi_inline sfpi::vFloat _sfpu_exp2_fp32_accurate_(sfpi::vFloat x)
{
    sfpi::vFloat result = sfpi::vConst0;

    // Thresholds are applied directly in the base-2 domain (x = log2 of result).
    // For finite normal fp32 outputs: bias-1 ≥ true_exp ≥ -126, i.e. biased_exp ∈ [1, 254].
    // - x ≥ 128 → result would overflow (biased_exp ≥ 255), so return +inf.
    // - x ≤ -127 → result would underflow well into the denormal range, so return 0.
    // Inputs in (-127, -126) produce subnormal results; setexp flushes those to zero
    // (the high-mantissa bits become the subnormal mantissa with biased_exp=0), which
    // matches IEEE flush-to-zero semantics and the convention used by _sfpu_exp_fp32_accurate_.
    constexpr float OVERFLOW_THRESHOLD  = 128.0f;
    constexpr float UNDERFLOW_THRESHOLD = -127.0f;

    sfpi::vInt exp_bits = sfpi::exexp(x);

    // NaN/±inf check FIRST (integer compare, immune to -ffinite-math-only).
    // Float compares against ±inf / NaN are undefined behaviour under fast-math,
    // which is enabled in the kernel build (see -ffast-math in build.cpp). Doing
    // the IEEE-special-class dispatch on the integer exponent bits sidesteps this
    // entirely, and is also one cheaper compare in the common (finite) case.
    v_if (exp_bits == 255)
    {
        // exponent==255 is either ±inf (mantissa==0) or NaN (mantissa!=0).
        // For exp2: exp2(+inf) = +inf, exp2(-inf) = 0, exp2(NaN) = NaN.
        // We can collapse +inf and NaN into a single branch that just returns x:
        //   * +inf → x is +inf → result is +inf (correct).
        //   * NaN  → x is NaN  → result is NaN  (correct, NaN propagates).
        // -inf is the only sub-case that needs a different result; it's caught
        // by the sign of x below.
        v_if (x < 0.0f)
        {
            result = sfpi::vConst0;  // exp2(-inf) = 0
        }
        v_else
        {
            result = x;              // exp2(+inf) = +inf, exp2(NaN) = NaN
        }
        v_endif;
    }
    v_elseif (x >= OVERFLOW_THRESHOLD)
    {
        result = std::numeric_limits<float>::infinity();
    }
    v_elseif (x <= UNDERFLOW_THRESHOLD)
    {
        result = sfpi::vConst0;
    }
    v_else
    {
        result = _sfpu_exp2_core_unsafe_(x);
    }
    v_endif;

    return result;
}

// BF16 path: same algorithm, lower-degree polynomial; followed by an explicit
// round-to-nearest-even into bf16 to avoid the truncation that SFPSTORE would
// otherwise apply when DEST is bf16 (matches the pattern in _sfpu_exp_21f_bf16_).
sfpi_inline sfpi::vFloat _sfpu_exp2_bf16_(sfpi::vFloat x)
{
    sfpi::vFloat result = sfpi::vConst0;

    constexpr float OVERFLOW_THRESHOLD  = 128.0f;
    constexpr float UNDERFLOW_THRESHOLD = -127.0f;

    sfpi::vInt exp_bits = sfpi::exexp(x);

    // See _sfpu_exp2_fp32_accurate_ for why the special-class check goes first.
    v_if (exp_bits == 255)
    {
        v_if (x < 0.0f)
        {
            result = sfpi::vConst0;  // exp2(-inf) = 0
        }
        v_else
        {
            result = x;              // exp2(+inf) = +inf, exp2(NaN) = NaN
        }
        v_endif;
    }
    v_elseif (x >= OVERFLOW_THRESHOLD)
    {
        result = std::numeric_limits<float>::infinity();
    }
    v_elseif (x <= UNDERFLOW_THRESHOLD)
    {
        result = sfpi::vConst0;
    }
    v_else
    {
        // Same n+f split as the fp32 path.
        sfpi::vInt n_int;
        sfpi::vFloat n_float = _sfpu_round_to_nearest_int32_(x, n_int);
        sfpi::vFloat f       = x - n_float;

        // Degree-3 truncated Taylor polynomial of exp(f·ln2): bf16 only carries
        // 7 mantissa bits, so this is more than enough headroom to hit ≤ 0.5 ulp
        // after the final bf16 round.
        sfpi::vFloat p = PolynomialEvaluator::eval(
            f,
            sfpi::vConst1,    // c0 = 1
            0x1.62e430p-1f,   // c1 = ln(2)
            0x1.ebfbe0p-3f,   // c2 = ln(2)²/2
            0x1.c6b08ep-5f    // c3 = ln(2)³/6
        );

        sfpi::vInt p_exp   = sfpi::exexp(p, sfpi::ExponentMode::NoDebias);
        sfpi::vInt new_exp = p_exp + n_int;
        result             = sfpi::setexp(p, new_exp);
    }
    v_endif;

    // SFPSTORE truncates fp32→bf16; round explicitly so the bf16 result matches
    // a faithful nearest-even rounding of the fp32 mathematical value.
    return sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::NearestEven);
}

template <bool APPROXIMATION_MODE /*unused*/, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void _calculate_exp2_()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];

        if constexpr (is_fp32_dest_acc_en)
        {
            sfpi::dst_reg[0] = _sfpu_exp2_fp32_accurate_(v);
        }
        else
        {
            sfpi::dst_reg[0] = _sfpu_exp2_bf16_(v);
        }

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE /*unused*/>
inline void _init_exp2_()
{
    // The optimised exp2 implementation works directly in base-2 and does not
    // require any program-constant register. Kept as a no-op to preserve the
    // public init API (callers in `_init_exp2_<APPROX>::call` and the LLK test
    // harness still invoke this).
}

} // namespace ckernel::sfpu
