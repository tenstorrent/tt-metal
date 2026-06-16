// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

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
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vUInt>(flag1) | sfpi::reinterpret<sfpi::vUInt>(flag2));
            }
            else
            {
                // greater_than_zero
                // flag1 = 0x3F80(1.0) if DST >= 0 else 0
                // flag2 = 0x3F80(1.0) if DST != 0 else 0
                // Do a bitwise And (flag1 & flag2) to get > condition.
                // flag2 >= 0 AND flag1 != 0 => DST is Greater than zero
                // Result will be either 0x0000(0.0) or 0x3F80(1.0)
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
sfpi_inline void apply_zero_comp(sfpi::vFloat& v, std::uint32_t exponent_size_8);

template <>
sfpi_inline void apply_zero_comp<SfpuType::equal_zero>(sfpi::vFloat& v, std::uint32_t)
{
    sfpi::vFloat r = 0.0f;
    v_if (_sfpu_is_fp16_zero_(v))
        r = 1.0f;
    v_endif;
    v = r;
}

template <>
sfpi_inline void apply_zero_comp<SfpuType::not_equal_zero>(sfpi::vFloat& v, std::uint32_t)
{
    sfpi::vFloat r = 1.0f;
    v_if (_sfpu_is_fp16_zero_(v))
        r = 0.0f;
    v_endif;
    v = r;
}

template <>
sfpi_inline void apply_zero_comp<SfpuType::less_than_zero>(sfpi::vFloat& v, std::uint32_t /*unused*/)
{
    sfpi::vFloat r = 1.0f;
    v_if (v >= 0.0f)
        r = 0.0f;
    v_endif;
    v = r;
}

template <>
sfpi_inline void apply_zero_comp<SfpuType::greater_than_equal_zero>(sfpi::vFloat& v, std::uint32_t /*unused*/)
{
    sfpi::vFloat r = 0.0f;
    v_if (v >= 0.0f)
        r = 1.0f;
    v_endif;
    v = r;
}

template <>
sfpi_inline void apply_zero_comp<SfpuType::greater_than_zero>(sfpi::vFloat& v, std::uint32_t /*unused*/)
{
    sfpi::vFloat r = 0.0f;
    v_if (v > 0.0f)
        r = 1.0f;
    v_endif;
    v = r;
}

template <>
sfpi_inline void apply_zero_comp<SfpuType::less_than_equal_zero>(sfpi::vFloat& v, std::uint32_t /*unused*/)
{
    sfpi::vFloat r = 1.0f;
    v_if (v > 0.0f)
        r = 0.0f;
    v_endif;
    v = r;
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
    sfpi::vInt r = 0;
    v_if (v == 0)
        r = 1;
    v_endif;
    v = r;
}

template <>
sfpi_inline void apply_zero_comp_int<SfpuType::not_equal_zero>(sfpi::vInt& v)
{
    sfpi::vInt r = 1;
    v_if (v == 0)
        r = 0;
    v_endif;
    v = r;
}

template <>
sfpi_inline void apply_zero_comp_int<SfpuType::less_than_zero>(sfpi::vInt& v)
{
    sfpi::vInt r = 0;
    v_if (v < 0)
        r = 1;
    v_endif;
    v = r;
}

template <>
sfpi_inline void apply_zero_comp_int<SfpuType::greater_than_zero>(sfpi::vInt& v)
{
    sfpi::vInt r = 0;
    v_if (v > 0)
        r = 1;
    v_endif;
    v = r;
}

template <>
sfpi_inline void apply_zero_comp_int<SfpuType::less_than_equal_zero>(sfpi::vInt& v)
{
    sfpi::vInt r = 0;
    v_if (v <= 0)
        r = 1;
    v_endif;
    v = r;
}

template <>
sfpi_inline void apply_zero_comp_int<SfpuType::greater_than_equal_zero>(sfpi::vInt& v)
{
    sfpi::vInt r = 0;
    v_if (v >= 0)
        r = 1;
    v_endif;
    v = r;
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
sfpi_inline void _calculate_zero_comp_int_()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vInt v = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
        apply_zero_comp_int<COMP_MODE>(v);
        sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = v;
        sfpi::dst_reg++;
    }
}

template <SfpuType COMP_MODE>
sfpi_inline void apply_unary_comp_int(sfpi::vInt& val, const sfpi::vInt& v, const int scalar);

template <>
sfpi_inline void apply_unary_comp_int<SfpuType::unary_ne>(sfpi::vInt& val, const sfpi::vInt& v, const int scalar)
{
    val = 0;
    v_if (v != scalar)
    {
        val = 1;
    }
    v_endif;
}

template <>
sfpi_inline void apply_unary_comp_int<SfpuType::unary_eq>(sfpi::vInt& val, const sfpi::vInt& v, const int scalar)
{
    val = 0;
    v_if (v == scalar)
    {
        val = 1;
    }
    v_endif;
}

template <>
sfpi_inline void apply_unary_comp_int<SfpuType::unary_gt>(sfpi::vInt& val, const sfpi::vInt& v, const int scalar)
{
    // Unfortunately the compiler doesn't generate correct compares
    // (#14598)
    sfpi::vInt s = scalar;

    val = 0;
    v_if ((v ^ s) >= 0)
    {
        // Same sign, compare is good
        v_if (v > s)
        {
            val = 1;
        }
        v_endif;
    }
    v_elseif (v >= 0)
    {
        val = 1;
    }
    v_endif;
}

template <>
sfpi_inline void apply_unary_comp_int<SfpuType::unary_lt>(sfpi::vInt& val, const sfpi::vInt& v, const int scalar)
{
    sfpi::vInt s = scalar;

    val = 0;
    v_if ((v ^ s) >= 0)
    {
        // Same sign, compare is good
        v_if (v < s)
        {
            val = 1;
        }
        v_endif;
    }
    v_elseif (v < 0)
    {
        val = 1;
    }
    v_endif;
}

template <>
sfpi_inline void apply_unary_comp_int<SfpuType::unary_ge>(sfpi::vInt& val, const sfpi::vInt& v, const int scalar)
{
    sfpi::vInt s = scalar;

    val = 0;
    v_if ((v ^ s) >= 0)
    {
        // Same sign, compare is good
        v_if (v >= s)
        {
            val = 1;
        }
        v_endif;
    }
    v_elseif (v >= 0)
    {
        val = 1;
    }
    v_endif;
}

template <>
sfpi_inline void apply_unary_comp_int<SfpuType::unary_le>(sfpi::vInt& val, const sfpi::vInt& v, const int scalar)
{
    sfpi::vInt s = scalar;

    val = 0;
    v_if ((v ^ s) >= 0)
    {
        // Same sign, compare is good
        v_if (v <= s)
        {
            val = 1;
        }
        v_endif;
    }
    v_elseif (v < 0)
    {
        val = 1;
    }
    v_endif;
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
sfpi_inline void _calculate_comp_unary_int_(int scalar)
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vInt v   = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
        sfpi::vInt val = 0;

        apply_unary_comp_int<COMP_MODE>(val, v, scalar);

        sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = val;
        sfpi::dst_reg++;
    }
}

template <SfpuType COMP_MODE>
sfpi_inline void apply_unary_comp_float(sfpi::vFloat& val, const sfpi::vFloat& v, const sfpi::vFloat& s);

template <>
sfpi_inline void apply_unary_comp_float<SfpuType::unary_eq>(sfpi::vFloat& val, const sfpi::vFloat& v, const sfpi::vFloat& s)
{
    val = 0.0f;
    v_if (v == s)
    {
        val = 1.0f;
    }
    v_endif;
}

template <>
sfpi_inline void apply_unary_comp_float<SfpuType::unary_ne>(sfpi::vFloat& val, const sfpi::vFloat& v, const sfpi::vFloat& s)
{
    val = 1.0f;
    v_if (v == s)
    {
        val = 0.0f;
    }
    v_endif;
}

template <>
sfpi_inline void apply_unary_comp_float<SfpuType::unary_gt>(sfpi::vFloat& val, const sfpi::vFloat& v, const sfpi::vFloat& s)
{
    val = 0.0f;
    v_if (v > s)
    {
        val = 1.0f;
    }
    v_endif;
}

template <>
sfpi_inline void apply_unary_comp_float<SfpuType::unary_lt>(sfpi::vFloat& val, const sfpi::vFloat& v, const sfpi::vFloat& s)
{
    val = 0.0f;
    v_if (v < s)
    {
        val = 1.0f;
    }
    v_endif;
}

template <>
sfpi_inline void apply_unary_comp_float<SfpuType::unary_ge>(sfpi::vFloat& val, const sfpi::vFloat& v, const sfpi::vFloat& s)
{
    val = 0.0f;
    v_if (v >= s)
    {
        val = 1.0f;
    }
    v_endif;
}

template <>
sfpi_inline void apply_unary_comp_float<SfpuType::unary_le>(sfpi::vFloat& val, const sfpi::vFloat& v, const sfpi::vFloat& s)
{
    val = 0.0f;
    v_if (v <= s)
    {
        val = 1.0f;
    }
    v_endif;
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
sfpi_inline void _calculate_comp_unary_(std::uint32_t value)
{
    sfpi::vFloat s = value;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
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
