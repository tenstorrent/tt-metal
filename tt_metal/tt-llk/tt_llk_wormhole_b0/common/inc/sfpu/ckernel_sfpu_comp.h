// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_is_fp16_zero.h"
#include "llk_sfpu_types.h"
#include "sfpi.h"

namespace
{
constexpr std::uint32_t ONE  = 1;
constexpr std::uint32_t ZERO = 0;
} // namespace

namespace ckernel
{
namespace sfpu
{

sfpi_inline void _calculate_comp_init_flag_(bool check, sfpi::vFloat& flag1, sfpi::vFloat& flag2, float init)
{
    flag1 = init;
    if (check)
    {
        flag2 = init;
    }
}

template <bool APPROXIMATION_MODE, bool invert_output, bool check_zero, bool second_check, bool is_less_than_equal_zero, int ITERATIONS>
inline void _calculate_comp_(const int iterations, std::uint32_t exponent_size_8)
{
    constexpr float output_0 = invert_output ? 0.0f : 1.0f;
    constexpr float output_1 = invert_output ? 1.0f : 0.0f;

    for (int d = ZERO; d < iterations; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat flag1, flag2;
        if constexpr (check_zero)
        {
            v_if (_sfpu_is_fp16_zero_(v, exponent_size_8))
            {
                _calculate_comp_init_flag_(second_check, flag1, flag2, output_0);
            }
            v_else
            {
                _calculate_comp_init_flag_(second_check, flag1, flag2, output_1);
            }
            v_endif;
        }
        else
        {
            v_if (v < 0.0F)
            {
                _calculate_comp_init_flag_(second_check, flag1, flag2, output_0);
            }
            v_else
            {
                _calculate_comp_init_flag_(second_check, flag1, flag2, output_1);
            }
            v_endif;
        }

        sfpi::vFloat result;
        if constexpr (second_check)
        {
            if constexpr (is_less_than_equal_zero)
            {
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vUInt>(flag1) | sfpi::reinterpret<sfpi::vUInt>(flag2));
            }
            else
            {
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vUInt>(flag1) & sfpi::reinterpret<sfpi::vUInt>(flag2));
            }
        }
        else
        {
            result = flag1;
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <SfpuType COMP_MODE>
inline void apply_zero_comp(sfpi::vFloat& v, std::uint32_t exponent_size_8);

template <>
inline void apply_zero_comp<SfpuType::equal_zero>(sfpi::vFloat& v, std::uint32_t exponent_size_8)
{
    v_if (_sfpu_is_fp16_zero_(v, exponent_size_8))
    {
        v = ONE;
    }
    v_else
    {
        v = ZERO;
    }
    v_endif;
}

template <>
inline void apply_zero_comp<SfpuType::not_equal_zero>(sfpi::vFloat& v, std::uint32_t exponent_size_8)
{
    v_if (_sfpu_is_fp16_zero_(v, exponent_size_8))
    {
        v = ZERO;
    }
    v_else
    {
        v = ONE;
    }
    v_endif;
}

template <>
inline void apply_zero_comp<SfpuType::less_than_zero>(sfpi::vFloat& v, std::uint32_t /*unused*/)
{
    v_if (v >= ZERO)
    {
        v = ZERO;
    }
    v_else
    {
        v = ONE;
    }
    v_endif;
}

template <>
inline void apply_zero_comp<SfpuType::greater_than_equal_zero>(sfpi::vFloat& v, std::uint32_t /*unused*/)
{
    v_if (v >= ZERO)
    {
        v = ONE;
    }
    v_else
    {
        v = ZERO;
    }
    v_endif;
}

template <>
inline void apply_zero_comp<SfpuType::greater_than_zero>(sfpi::vFloat& v, std::uint32_t /*unused*/)
{
    v_if (v > ZERO)
    {
        v = ONE;
    }
    v_else
    {
        v = ZERO;
    }
    v_endif;
}

template <>
inline void apply_zero_comp<SfpuType::less_than_equal_zero>(sfpi::vFloat& v, std::uint32_t /*unused*/)
{
    v_if (v > ZERO)
    {
        v = ZERO;
    }
    v_else
    {
        v = ONE;
    }
    v_endif;
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void _calculate_zero_comp_(std::uint32_t exponent_size_8)
{
    for (int d = ZERO; d < ITERATIONS; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];
        apply_zero_comp<COMP_MODE>(v, exponent_size_8);
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

// ---- Integer comparison helpers ----

// Branch-free sign extraction for sign-magnitude vInt
// Returns: 0 for positive, 1 for negative (as vUInt)
sfpi_inline sfpi::vUInt _sign_bit_(sfpi::vInt v)
{
    // In sign-magnitude, MSB is the sign bit
    // Reinterpret as vUInt to get the raw bits
    return sfpi::reinterpret<sfpi::vUInt>(v) >> 31;
}

// ---- Zero-comparison for integer types ----

template <SfpuType COMP_MODE>
inline void apply_zero_comp_int(sfpi::vInt& v);

template <>
inline void apply_zero_comp_int<SfpuType::equal_zero>(sfpi::vInt& v)
{
    v_if (v == ZERO)
    {
        v = ONE;
    }
    v_else
    {
        v = ZERO;
    }
    v_endif;
}

template <>
inline void apply_zero_comp_int<SfpuType::not_equal_zero>(sfpi::vInt& v)
{
    v_if (v == ZERO)
    {
        v = ZERO;
    }
    v_else
    {
        v = ONE;
    }
    v_endif;
}

template <>
inline void apply_zero_comp_int<SfpuType::less_than_zero>(sfpi::vInt& v)
{
    v_if (v < ZERO)
    {
        v = ONE;
    }
    v_else
    {
        v = ZERO;
    }
    v_endif;
}

template <>
inline void apply_zero_comp_int<SfpuType::greater_than_zero>(sfpi::vInt& v)
{
    v_if (v > ZERO)
    {
        v = ONE;
    }
    v_else
    {
        v = ZERO;
    }
    v_endif;
}

template <>
inline void apply_zero_comp_int<SfpuType::less_than_equal_zero>(sfpi::vInt& v)
{
    v_if (v <= ZERO)
    {
        v = ONE;
    }
    v_else
    {
        v = ZERO;
    }
    v_endif;
}

template <>
inline void apply_zero_comp_int<SfpuType::greater_than_equal_zero>(sfpi::vInt& v)
{
    v_if (v >= ZERO)
    {
        v = ONE;
    }
    v_else
    {
        v = ZERO;
    }
    v_endif;
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void _calculate_zero_comp_int_()
{
    for (int d = ZERO; d < ITERATIONS; d++)
    {
        sfpi::vInt v = sfpi::dst_reg[0];
        apply_zero_comp_int<COMP_MODE>(v);
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

// ---- Unary (scalar) comparison for integer types ----
// Optimized for WH: reduces branching from 3-4 levels to 2 levels
// by handling sign checks first, then falling through to direct comparison

template <SfpuType COMP_MODE>
inline void apply_unary_int_comp(sfpi::vInt& v, int scalar, sfpi::vInt& out_val);

// a[i] > scalar — optimized: flat sign check + direct magnitude comparison
template <>
inline void apply_unary_int_comp<SfpuType::unary_gt>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val)
{
    const sfpi::vInt s = scalar;
    v_if (v >= ZERO && s < ZERO)
    {
        out_val = ONE;
    }
    v_elseif (v < ZERO && s >= ZERO)
    {
        out_val = ZERO;
    }
    v_elseif (v > s)
    {
        out_val = ONE;
    }
    v_endif;
}

// a[i] < scalar — optimized: flat sign check + direct magnitude comparison
template <>
inline void apply_unary_int_comp<SfpuType::unary_lt>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val)
{
    const sfpi::vInt s = scalar;
    v_if (v >= ZERO && s < ZERO)
    {
        out_val = ZERO;
    }
    v_elseif (v < ZERO && s >= ZERO)
    {
        out_val = ONE;
    }
    v_elseif (v < s)
    {
        out_val = ONE;
    }
    v_endif;
}

// a[i] >= scalar — optimized
template <>
inline void apply_unary_int_comp<SfpuType::unary_ge>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val)
{
    const sfpi::vInt s = scalar;
    v_if (v >= ZERO && s < ZERO)
    {
        out_val = ONE;
    }
    v_elseif (v < ZERO && s >= ZERO)
    {
        out_val = ZERO;
    }
    v_elseif (v >= s)
    {
        out_val = ONE;
    }
    v_endif;
}

// a[i] <= scalar — optimized
template <>
inline void apply_unary_int_comp<SfpuType::unary_le>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val)
{
    const sfpi::vInt s = scalar;
    v_if (v < ZERO && s >= ZERO)
    {
        out_val = ONE;
    }
    v_elseif (v >= ZERO && s < ZERO)
    {
        out_val = ZERO;
    }
    v_elseif (v <= s)
    {
        out_val = ONE;
    }
    v_else
    {
        out_val = ZERO;
    }
    v_endif;
}

// a[i] == scalar
template <>
inline void apply_unary_int_comp<SfpuType::unary_eq>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val)
{
    v_if (v == scalar)
    {
        out_val = ONE;
    }
    v_endif;
}

// a[i] != scalar
template <>
inline void apply_unary_int_comp<SfpuType::unary_ne>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val)
{
    v_if (v != scalar)
    {
        out_val = ONE;
    }
    v_endif;
}

// ---- Unary (scalar) comparison for unsigned integer types ----
// No sign handling needed — direct comparison only, ~2x faster

// uint32 > scalar
template <>
inline void apply_unary_int_comp<SfpuType::unary_max_uint32>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val)
{
    const sfpi::vInt s = scalar;
    v_if (v > s)
    {
        out_val = ONE;
    }
    v_endif;
}

// uint32 < scalar
template <>
inline void apply_unary_int_comp<SfpuType::unary_min_uint32>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val)
{
    const sfpi::vInt s = scalar;
    v_if (v < s)
    {
        out_val = ONE;
    }
    v_endif;
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void _calculate_comp_unary_int_(int scalar)
{
#pragma GCC unroll 8
    for (int d = ZERO; d < ITERATIONS; d++)
    {
        sfpi::vInt v   = sfpi::dst_reg[0];
        sfpi::vInt val = ZERO;

        apply_unary_int_comp<COMP_MODE>(v, scalar, val);

        sfpi::dst_reg[0] = val;
        sfpi::dst_reg++;
    }
}

// ---- Float comparison with scalar ----

template <SfpuType COMP_MODE>
inline void apply_unary_comp_float(sfpi::vFloat& val, const sfpi::vFloat& v, const sfpi::vFloat& s);

template <>
inline void apply_unary_comp_float<SfpuType::unary_eq>(sfpi::vFloat& val, const sfpi::vFloat& v, const sfpi::vFloat& s)
{
    v_if (v == s)
    {
        val = ONE;
    }
    v_else
    {
        val = ZERO;
    }
    v_endif;
}

template <>
inline void apply_unary_comp_float<SfpuType::unary_ne>(sfpi::vFloat& val, const sfpi::vFloat& v, const sfpi::vFloat& s)
{
    v_if (v == s)
    {
        val = ZERO;
    }
    v_else
    {
        val = ONE;
    }
    v_endif;
}

template <>
inline void apply_unary_comp_float<SfpuType::unary_gt>(sfpi::vFloat& val, const sfpi::vFloat& v, const sfpi::vFloat& s)
{
    v_if (v > s)
    {
        val = ONE;
    }
    v_else
    {
        val = ZERO;
    }
    v_endif;
}

template <>
inline void apply_unary_comp_float<SfpuType::unary_lt>(sfpi::vFloat& val, const sfpi::vFloat& v, const sfpi::vFloat& s)
{
    v_if (v < s)
    {
        val = ONE;
    }
    v_else
    {
        val = ZERO;
    }
    v_endif;
}

template <>
inline void apply_unary_comp_float<SfpuType::unary_ge>(sfpi::vFloat& val, const sfpi::vFloat& v, const sfpi::vFloat& s)
{
    v_if (v >= s)
    {
        val = ONE;
    }
    v_else
    {
        val = ZERO;
    }
    v_endif;
}

template <>
inline void apply_unary_comp_float<SfpuType::unary_le>(sfpi::vFloat& val, const sfpi::vFloat& v, const sfpi::vFloat& s)
{
    v_if (v <= s)
    {
        val = ONE;
    }
    v_else
    {
        val = ZERO;
    }
    v_endif;
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void _calculate_comp_unary_(std::uint32_t value)
{
    sfpi::vFloat s = value;

#pragma GCC unroll 8
    for (int d = ZERO; d < ITERATIONS; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat val;

        apply_unary_comp_float<COMP_MODE>(val, v, s);

        sfpi::dst_reg[0] = val;
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
