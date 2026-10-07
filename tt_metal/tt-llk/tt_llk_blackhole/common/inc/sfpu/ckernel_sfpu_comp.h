// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "ckernel_sfpu_is_fp16_zero.h"
#include "llk_sfpu_types.h"
#include "sfpi.h"

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
sfpi_inline void _calculate_comp_(const int iterations, std::uint32_t exponent_size_8)
{
    // output_0 and output_1 hold the outputs use use when a zero or negative check is true/false.
    // False = 0.0 = kCONST_0 (5/8-bit exponent format)
    // True  = 1.0 = kCONST_1_FP16B (8-bit exponent format)
    // SFPU uses 8-bit exponent in operations so loading these constants in 8-bit exponent format.
    // Although a command flag can tell SFPU to re-bias a 5-bit exponent to 8-bit, we are loading 8-bit
    // exponent and telling SFPU to not add any bias to these constants.
    constexpr float output_0 = invert_output ? 0.0f : 1.0f;
    constexpr float output_1 = invert_output ? 1.0f : 0.0f;

    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat flag1, flag2;
        if constexpr (check_zero)
        {
            v_if (_sfpu_is_fp16_zero_(v))
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
            // less_than_equal_zero
            // flag1 = 0x3F80(1.0) if DST < 0 else 0
            // flag2 = 0x3F80(1.0) if DST == 0 else 0
            // Do a bitwise Or (flag1 | flag2) to get <= condition.
            // flag1 < 0 OR flag2 == 0 => DST is Less than or Equal to zero.
            // Result will be either 0x0000(0.0) or 0x3F80(1.0)
            if constexpr (is_less_than_equal_zero)
            {
                result = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(flag1) | sfpi::as<sfpi::vUInt>(flag2));
            }
            else
            {
                // greater_than_zero
                // flag1 = 0x3F80(1.0) if DST >= 0 else 0
                // flag2 = 0x3F80(1.0) if DST != 0 else 0
                // Do a bitwise And (flag1 & flag2) to get > condition.
                // flag2 >= 0 AND flag1 != 0 => DST is Greater than zero
                // Result will be either 0x0000(0.0) or 0x3F80(1.0)
                result = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(flag1) & sfpi::as<sfpi::vUInt>(flag2));
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
sfpi_inline void apply_zero_comp(sfpi::vFloat& v, std::uint32_t exponent_size_8);

template <>
sfpi_inline void apply_zero_comp<SfpuType::equal_zero>(sfpi::vFloat& v, std::uint32_t)
{
    v_if (_sfpu_is_fp16_zero_(v))
    {
        v = 1.0f;
    }
    v_else
    {
        v = 0.0f;
    }
    v_endif;
}

template <>
sfpi_inline void apply_zero_comp<SfpuType::not_equal_zero>(sfpi::vFloat& v, std::uint32_t)
{
    v_if (_sfpu_is_fp16_zero_(v))
    {
        v = 0.0f;
    }
    v_else
    {
        v = 1.0f;
    }
    v_endif;
}

template <>
sfpi_inline void apply_zero_comp<SfpuType::less_than_zero>(sfpi::vFloat& v, std::uint32_t /*unused*/)
{
    v_if (v >= 0.0f)
    {
        v = 0.0f;
    }
    v_else
    {
        v = 1.0f;
    }
    v_endif;
}

template <>
sfpi_inline void apply_zero_comp<SfpuType::greater_than_equal_zero>(sfpi::vFloat& v, std::uint32_t /*unused*/)
{
    v_if (v >= 0.0f)
    {
        v = 1.0f;
    }
    v_else
    {
        v = 0.0f;
    }
    v_endif;
}

template <>
sfpi_inline void apply_zero_comp<SfpuType::greater_than_zero>(sfpi::vFloat& v, std::uint32_t /*unused*/)
{
    v_if (v > 0.0f)
    {
        v = 1.0f;
    }
    v_else
    {
        v = 0.0f;
    }
    v_endif;
}

template <>
sfpi_inline void apply_zero_comp<SfpuType::less_than_equal_zero>(sfpi::vFloat& v, std::uint32_t /*unused*/)
{
    v_if (v > 0.0f)
    {
        v = 0.0f;
    }
    v_else
    {
        v = 1.0f;
    }
    v_endif;
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
sfpi_inline void _calculate_zero_comp_(std::uint32_t exponent_size_8)
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];
        apply_zero_comp<COMP_MODE>(v, exponent_size_8);
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

template <SfpuType COMP_MODE>
sfpi_inline void apply_zero_comp_int(sfpi::vInt& v);

template <>
sfpi_inline void apply_zero_comp_int<SfpuType::equal_zero>(sfpi::vInt& v)
{
    v_if (v == 0)
    {
        v = 1;
    }
    v_else
    {
        v = 0;
    }
    v_endif;
}

template <>
sfpi_inline void apply_zero_comp_int<SfpuType::not_equal_zero>(sfpi::vInt& v)
{
    v_if (v == 0)
    {
        v = 0;
    }
    v_else
    {
        v = 1;
    }
    v_endif;
}

template <>
sfpi_inline void apply_zero_comp_int<SfpuType::less_than_zero>(sfpi::vInt& v)
{
    v_if (v < 0)
    {
        v = 1;
    }
    v_else
    {
        v = 0;
    }
    v_endif;
}

template <>
sfpi_inline void apply_zero_comp_int<SfpuType::greater_than_zero>(sfpi::vInt& v)
{
    v_if (v > 0)
    {
        v = 1;
    }
    v_else
    {
        v = 0;
    }
    v_endif;
}

template <>
sfpi_inline void apply_zero_comp_int<SfpuType::less_than_equal_zero>(sfpi::vInt& v)
{
    v_if (v <= 0)
    {
        v = 1;
    }
    v_else
    {
        v = 0;
    }
    v_endif;
}

template <>
sfpi_inline void apply_zero_comp_int<SfpuType::greater_than_equal_zero>(sfpi::vInt& v)
{
    v_if (v >= 0)
    {
        v = 1;
    }
    v_else
    {
        v = 0;
    }
    v_endif;
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
sfpi_inline void _calculate_zero_comp_int_()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vInt v = sfpi::dst_reg[0];
        apply_zero_comp_int<COMP_MODE>(v);
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

template <SfpuType COMP_MODE>
sfpi_inline void apply_unary_int_comp(sfpi::vInt& v, int scalar, sfpi::vInt& out_val);

// a[i] != scalar
template <>
sfpi_inline void apply_unary_int_comp<SfpuType::unary_ne>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val)
{
    v_if (v != scalar)
    {
        out_val = 1;
    }
    v_endif;
}

// a[i] == scalar
template <>
sfpi_inline void apply_unary_int_comp<SfpuType::unary_eq>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val)
{
    v_if (v == scalar)
    {
        out_val = 1;
    }
    v_endif;
}

// The ordered scalar compares are evaluated on the sign bit, as in #58193 (@ldjurovicTT): no condition codes per row.
sfpi_inline sfpi::vInt _int_sign_bit_(const sfpi::vInt x)
{
    return sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(x) >> 31);
}

// v < s is sign(v | (v - s)) for s >= 0 and sign(v & (v - s)) for s < 0; v - s decides only where v and s share a sign.
template <bool SCALAR_NEGATIVE, int ITERATIONS>
sfpi_inline void _comp_unary_int_lt_rows_(const sfpi::vInt s)
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        const sfpi::vInt v    = sfpi::dst_reg[0];
        const sfpi::vInt diff = v - s;
        if constexpr (SCALAR_NEGATIVE)
        {
            sfpi::dst_reg[0] = _int_sign_bit_(v & diff);
        }
        else
        {
            sfpi::dst_reg[0] = _int_sign_bit_(v | diff);
        }
        sfpi::dst_reg++;
    }
}

// v > s is sign(~v & (s - v)) for s >= 0 and sign(~v | (s - v)) for s < 0.
template <bool SCALAR_NEGATIVE, int ITERATIONS>
sfpi_inline void _comp_unary_int_gt_rows_(const sfpi::vInt s)
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        const sfpi::vInt v     = sfpi::dst_reg[0];
        const sfpi::vInt not_v = ~v;
        const sfpi::vInt diff  = s - v;
        if constexpr (SCALAR_NEGATIVE)
        {
            sfpi::dst_reg[0] = _int_sign_bit_(not_v | diff);
        }
        else
        {
            sfpi::dst_reg[0] = _int_sign_bit_(not_v & diff);
        }
        sfpi::dst_reg++;
    }
}

template <bool IS_GT, int ITERATIONS>
sfpi_inline void _comp_unary_int_ordered_(const int scalar)
{
    const sfpi::vInt s = scalar;
    if (scalar < 0)
    {
        if constexpr (IS_GT)
        {
            _comp_unary_int_gt_rows_<true, ITERATIONS>(s);
        }
        else
        {
            _comp_unary_int_lt_rows_<true, ITERATIONS>(s);
        }
    }
    else
    {
        if constexpr (IS_GT)
        {
            _comp_unary_int_gt_rows_<false, ITERATIONS>(s);
        }
        else
        {
            _comp_unary_int_lt_rows_<false, ITERATIONS>(s);
        }
    }
}

// Every int32 satisfies v <= INT_MAX and v >= INT_MIN.
template <int ITERATIONS>
sfpi_inline void _comp_unary_int_all_true_()
{
    const sfpi::vInt one = 1;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::dst_reg[0] = one;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
sfpi_inline void _calculate_comp_unary_int_(int scalar)
{
    if constexpr (COMP_MODE == SfpuType::unary_lt)
    {
        _comp_unary_int_ordered_<false, ITERATIONS>(scalar);
    }
    else if constexpr (COMP_MODE == SfpuType::unary_gt)
    {
        _comp_unary_int_ordered_<true, ITERATIONS>(scalar);
    }
    else if constexpr (COMP_MODE == SfpuType::unary_le)
    {
        // v <= s is v < s + 1
        if (scalar == std::numeric_limits<std::int32_t>::max())
        {
            _comp_unary_int_all_true_<ITERATIONS>();
        }
        else
        {
            _comp_unary_int_ordered_<false, ITERATIONS>(scalar + 1);
        }
    }
    else if constexpr (COMP_MODE == SfpuType::unary_ge)
    {
        // v >= s is v > s - 1
        if (scalar == std::numeric_limits<std::int32_t>::min())
        {
            _comp_unary_int_all_true_<ITERATIONS>();
        }
        else
        {
            _comp_unary_int_ordered_<true, ITERATIONS>(scalar - 1);
        }
    }
    else
    {
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            sfpi::vInt v   = sfpi::dst_reg[0];
            sfpi::vInt val = 0;

            apply_unary_int_comp<COMP_MODE>(v, scalar, val);

            sfpi::dst_reg[0] = val;
            sfpi::dst_reg++;
        }
    }
}

template <SfpuType COMP_MODE>
sfpi_inline void apply_unary_float_comp(sfpi::vFloat v, sfpi::vFloat scalar, sfpi::vFloat& out_val);

// a[i] == scalar
template <>
sfpi_inline void apply_unary_float_comp<SfpuType::unary_eq>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val)
{
    v_if (v == s)
    {
        out_val = 1.0f;
    }
    v_else
    {
        out_val = 0.0f;
    }
    v_endif;
}

// a[i] != scalar
template <>
sfpi_inline void apply_unary_float_comp<SfpuType::unary_ne>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val)
{
    v_if (v == s)
    {
        out_val = 0.0f;
    }
    v_else
    {
        out_val = 1.0f;
    }
    v_endif;
}

// a[i] > scalar
template <>
sfpi_inline void apply_unary_float_comp<SfpuType::unary_gt>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val)
{
    v_if (v > s)
    {
        out_val = 1.0f;
    }
    v_else
    {
        out_val = 0.0f;
    }
    v_endif;
}

// a[i] < scalar
template <>
sfpi_inline void apply_unary_float_comp<SfpuType::unary_lt>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val)
{
    v_if (v < s)
    {
        out_val = 1.0f;
    }
    v_else
    {
        out_val = 0.0f;
    }
    v_endif;
}

// a[i] >= scalar
template <>
sfpi_inline void apply_unary_float_comp<SfpuType::unary_ge>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val)
{
    v_if (v >= s)
    {
        out_val = 1.0f;
    }
    v_else
    {
        out_val = 0.0f;
    }
    v_endif;
}

// a[i] <= scalar
template <>
sfpi_inline void apply_unary_float_comp<SfpuType::unary_le>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val)
{
    v_if (v <= s)
    {
        out_val = 1.0f;
    }
    v_else
    {
        out_val = 0.0f;
    }
    v_endif;
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
sfpi_inline void _calculate_comp_unary_(std::uint32_t value)
{
    const sfpi::vFloat s = value;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat v   = sfpi::dst_reg[0];
        sfpi::vFloat val = 0.0f;

        apply_unary_float_comp<COMP_MODE>(v, s, val);

        sfpi::dst_reg[0] = val;
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
