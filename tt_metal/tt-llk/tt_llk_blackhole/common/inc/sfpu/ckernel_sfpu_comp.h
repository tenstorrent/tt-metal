// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_is_fp16_zero.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

enum class ZeroCompMode
{
    EqZ,
    NeZ,
    LtZ,
    GeZ,
    GtZ,
    LeZ
};

enum class UnaryCompMode
{
    Ne,
    Eq,
    Gt,
    Lt,
    Ge,
    Le
};

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

template <ZeroCompMode COMP_MODE>
sfpi_inline void apply_zero_comp(sfpi::vFloat& v, std::uint32_t exponent_size_8);

template <>
sfpi_inline void apply_zero_comp<ZeroCompMode::EqZ>(sfpi::vFloat& v, std::uint32_t)
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
sfpi_inline void apply_zero_comp<ZeroCompMode::NeZ>(sfpi::vFloat& v, std::uint32_t)
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
sfpi_inline void apply_zero_comp<ZeroCompMode::LtZ>(sfpi::vFloat& v, std::uint32_t /*unused*/)
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
sfpi_inline void apply_zero_comp<ZeroCompMode::GeZ>(sfpi::vFloat& v, std::uint32_t /*unused*/)
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
sfpi_inline void apply_zero_comp<ZeroCompMode::GtZ>(sfpi::vFloat& v, std::uint32_t /*unused*/)
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
sfpi_inline void apply_zero_comp<ZeroCompMode::LeZ>(sfpi::vFloat& v, std::uint32_t /*unused*/)
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

template <bool APPROXIMATION_MODE, ZeroCompMode COMP_MODE, int ITERATIONS = 8>
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

template <ZeroCompMode COMP_MODE>
sfpi_inline void apply_zero_comp_int(sfpi::vInt& v);

template <>
sfpi_inline void apply_zero_comp_int<ZeroCompMode::EqZ>(sfpi::vInt& v)
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
sfpi_inline void apply_zero_comp_int<ZeroCompMode::NeZ>(sfpi::vInt& v)
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
sfpi_inline void apply_zero_comp_int<ZeroCompMode::LtZ>(sfpi::vInt& v)
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
sfpi_inline void apply_zero_comp_int<ZeroCompMode::GtZ>(sfpi::vInt& v)
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
sfpi_inline void apply_zero_comp_int<ZeroCompMode::LeZ>(sfpi::vInt& v)
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
sfpi_inline void apply_zero_comp_int<ZeroCompMode::GeZ>(sfpi::vInt& v)
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

template <bool APPROXIMATION_MODE, ZeroCompMode COMP_MODE, int ITERATIONS = 8>
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

template <UnaryCompMode COMP_MODE>
sfpi_inline void apply_unary_int_comp(sfpi::vInt& v, int scalar, sfpi::vInt& out_val);

// a[i] != scalar
template <>
sfpi_inline void apply_unary_int_comp<UnaryCompMode::Ne>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val)
{
    v_if (v != scalar)
    {
        out_val = 1;
    }
    v_endif;
}

// a[i] == scalar
template <>
sfpi_inline void apply_unary_int_comp<UnaryCompMode::Eq>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val)
{
    v_if (v == scalar)
    {
        out_val = 1;
    }
    v_endif;
}

// a[i] > scalar
template <>
sfpi_inline void apply_unary_int_comp<UnaryCompMode::Gt>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val)
{
    const sfpi::vInt s = scalar;
    v_if (v >= 0 && s < 0)
    {
        out_val = 1;
    }
    v_elseif (v < 0 && s >= 0)
    {
        out_val = 0;
    }
    v_elseif (v > s)
    {
        out_val = 1;
    }
    v_endif;
}

// a[i] < scalar
template <>
sfpi_inline void apply_unary_int_comp<UnaryCompMode::Lt>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val)
{
    const sfpi::vInt s = scalar;
    v_if (v >= 0 && s < 0)
    {
        out_val = 0;
    }
    v_elseif (v < 0 && s >= 0)
    {
        out_val = 1;
    }
    v_elseif (v < s)
    {
        out_val = 1;
    }
    v_endif;
}

// a[i] >= scalar
template <>
sfpi_inline void apply_unary_int_comp<UnaryCompMode::Ge>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val)
{
    const sfpi::vInt s = scalar;
    v_if (v >= 0 && s < 0)
    {
        out_val = 1;
    }
    v_elseif (v < 0 && s >= 0)
    {
        out_val = 0;
    }
    v_elseif (v >= s)
    {
        out_val = 1;
    }
    v_endif;
}

// a[i] <= scalar
template <>
sfpi_inline void apply_unary_int_comp<UnaryCompMode::Le>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val)
{
    const sfpi::vInt s = scalar;
    v_if (v < 0 && s >= 0)
    {
        out_val = 1;
    }
    v_elseif (v >= 0 && s < 0)
    {
        out_val = 0;
    }
    v_elseif (v <= s)
    {
        out_val = 1;
    }
    v_else
    {
        out_val = 0;
    }
    v_endif;
}

template <bool APPROXIMATION_MODE, UnaryCompMode COMP_MODE, int ITERATIONS = 8>
sfpi_inline void _calculate_comp_unary_int_(int scalar)
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

template <UnaryCompMode COMP_MODE>
sfpi_inline void apply_unary_float_comp(sfpi::vFloat v, sfpi::vFloat scalar, sfpi::vFloat& out_val);

// a[i] == scalar
template <>
sfpi_inline void apply_unary_float_comp<UnaryCompMode::Eq>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val)
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
sfpi_inline void apply_unary_float_comp<UnaryCompMode::Ne>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val)
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
sfpi_inline void apply_unary_float_comp<UnaryCompMode::Gt>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val)
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
sfpi_inline void apply_unary_float_comp<UnaryCompMode::Lt>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val)
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
sfpi_inline void apply_unary_float_comp<UnaryCompMode::Ge>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val)
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
sfpi_inline void apply_unary_float_comp<UnaryCompMode::Le>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val)
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

template <bool APPROXIMATION_MODE, UnaryCompMode COMP_MODE, int ITERATIONS = 8>
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
