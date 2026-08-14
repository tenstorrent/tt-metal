// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "sfpi.h"

namespace ckernel::sfpu::compat
{

// Quasar SFPI compiler comparison workaround. The affected compiler lowers
// vFloat <, >=, == and != through SFPSETCC with
// Imm12[11]=0, which makes Quasar interpret the FP32 difference as INT32.  The
// native > and <= forms select the expected FP32 comparison path, so express all
// four affected predicates in terms of only those two known-good operators.
// Keep these wrappers until the compiler sets SFPSETCC.Imm12[11] correctly.
sfpi_inline sfpi::vBool fp_lt(sfpi::vFloat lhs, sfpi::vFloat rhs)
{
    return rhs > lhs;
}

sfpi_inline sfpi::vBool fp_ge(sfpi::vFloat lhs, sfpi::vFloat rhs)
{
    return rhs <= lhs;
}

sfpi_inline sfpi::vBool fp_eq(sfpi::vFloat lhs, sfpi::vFloat rhs)
{
    return (lhs <= rhs) && (rhs <= lhs);
}

sfpi_inline sfpi::vBool fp_ne(sfpi::vFloat lhs, sfpi::vFloat rhs)
{
    return !fp_eq(lhs, rhs);
}

sfpi_inline sfpi::vBool fp_lt(sfpi::vFloat lhs, float rhs)
{
    return sfpi::vFloat(rhs) > lhs;
}

sfpi_inline sfpi::vBool fp_ge(sfpi::vFloat lhs, float rhs)
{
    return sfpi::vFloat(rhs) <= lhs;
}

// Equality with zero has a cheaper exact implementation than the general
// two-sided ordered comparison. Shift away the IEEE sign bit so +0 and -0
// both compare equal while every non-zero value (including NaN/Inf) does not.
// Besides matching IEEE zero semantics, this avoids extra live vFloat values
// in register-heavy kernels such as the accurate FP32 power implementation.
sfpi_inline sfpi::vBool fp_is_zero(sfpi::vFloat value)
{
    return (sfpi::as<sfpi::vUInt>(value) << 1) == 0U;
}

sfpi_inline sfpi::vBool fp_eq(sfpi::vFloat lhs, float rhs)
{
    const sfpi::vFloat rhs_v = rhs;
    return (lhs <= rhs_v) && (rhs_v <= lhs);
}

sfpi_inline sfpi::vBool fp_ne(sfpi::vFloat lhs, float rhs)
{
    return !fp_eq(lhs, rhs);
}

sfpi_inline sfpi::vFloat fp_max(sfpi::vFloat lhs, sfpi::vFloat rhs)
{
    sfpi::vFloat result = rhs;
    v_if (lhs > rhs)
    {
        result = lhs;
    }
    v_endif;
    return result;
}

sfpi_inline sfpi::vFloat fp_min(sfpi::vFloat lhs, sfpi::vFloat rhs)
{
    sfpi::vFloat result = rhs;
    v_if (lhs <= rhs)
    {
        result = lhs;
    }
    v_endif;
    return result;
}

sfpi_inline sfpi::vFloat fp_clamp(sfpi::vFloat value, sfpi::vFloat lower, sfpi::vFloat upper)
{
    return fp_min(fp_max(value, lower), upper);
}

} // namespace ckernel::sfpu::compat

// Keep the comparison definitions above these architecture headers.  Quasar's
// native log/reciprocal implementations include this file to use them, so this
// ordering also makes the include cycle well-formed.
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu
{

// Source-compatible names used by architecture-neutral Blackhole SFPI
// kernels.  Quasar's native primitives use leading-underscore spellings.
template <bool APPROXIMATE = false, [[maybe_unused]] bool save_reg = true>
sfpi_inline sfpi::vFloat sfpu_reciprocal(const sfpi::vFloat input)
{
    return _sfpu_reciprocal_<APPROXIMATE ? 0 : 2>(input);
}

template <bool APPROXIMATE = false>
sfpi_inline void sfpu_reciprocal_init()
{
    _init_reciprocal_<APPROXIMATE>();
}

sfpi_inline sfpi::vFloat _sfpu_exp_fp32_accurate_unsafe_(sfpi::vFloat input)
{
    return _sfpu_exp_fp32_accurate_(input);
}

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_21f_bf16_unsafe_(sfpi::vFloat input)
{
    sfpi::vFloat result = _sfpu_exp_fp32_accurate_(input);
    if constexpr (!is_fp32_dest_acc_en)
    {
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
    }
    return result;
}

// BH's activation helper is a clamp to [0, threshold].  Spell the lower
// comparison as 0 > result so Quasar never lowers the broken vFloat '<'.
sfpi_inline sfpi::vFloat _relu_max_body_(sfpi::vFloat input, sfpi::vFloat threshold)
{
    sfpi::vFloat result = input;
    v_if (result > threshold)
    {
        result = threshold;
    }
    v_endif;
    v_if (sfpi::vFloat(0.0f) > result)
    {
        result = 0.0f;
    }
    v_endif;
    return result;
}

// Legacy single-value log entry point used by erfinv.  Quasar's native
// _calculate_log_body_no_init_ has a raw vFloat == 0 comparison, so repeat its
// compact polynomial here and use only the comparison compatibility helpers.
template <[[maybe_unused]] bool FAST_APPROX, bool HAS_BASE_SCALING, [[maybe_unused]] bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat calculate_log_body(sfpi::vFloat base, const std::uint32_t log_base_scale_factor)
{
    sfpi::vFloat result = std::numeric_limits<float>::quiet_NaN();
    v_if (compat::fp_ge(base, 0.0f))
    {
        const sfpi::vFloat x       = sfpi::setexp(base, 127);
        sfpi::vFloat series_result = x * (x * (x * 0x2.44734p-4f - 0xd.e712ap-4f) + 0x2.4f5388p+0f) - 0x1.952992p+0f;

        const sfpi::vSMag exp   = sfpi::convert<sfpi::vSMag>(sfpi::exexp(base));
        const sfpi::vFloat expf = sfpi::convert<sfpi::vFloat>(exp, sfpi::RoundMode::Nearest);
        result                  = expf * 0.692871f + series_result;

        if constexpr (HAS_BASE_SCALING)
        {
            result *= sfpi::as<sfpi::vFloat>(sfpi::vUInt(log_base_scale_factor));
        }

        v_if (compat::fp_eq(base, 0.0f))
        {
            result = -std::numeric_limits<float>::infinity();
        }
        v_endif;
    }
    v_endif;
    return result;
}

template <bool APPROXIMATION_MODE, [[maybe_unused]] bool FAST_APPROX, [[maybe_unused]] bool is_fp32_dest_acc_en>
inline void log_init()
{
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstFloatPrgm0 = 0.692871f;
    sfpi::vConstFloatPrgm1 = 0.1058f;
    sfpi::vConstFloatPrgm2 = -0.7166f;
}

} // namespace ckernel::sfpu
