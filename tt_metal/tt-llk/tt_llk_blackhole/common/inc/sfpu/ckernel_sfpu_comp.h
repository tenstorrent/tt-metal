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

// ─── Integer compares ───────────────────────────────────────────────────────────────────────────
// Every integer compare below is evaluated arithmetically instead of through v_if/v_else: the
// verdict is formed in bit 31 (a sign) or in bit 5 (SFPLZ reports 32 for an all-zero word) and
// shifted down to an integer 0/1, so a row costs no condition-code traffic
// (SFPSETCC/SFPCOMPC/SFPENCC) and materialises no per-row constant. Inputs are two's-complement
// int32 in Dst (the ttnn convention; sfpi's DataLayout::I32, a raw load on Blackhole) and the
// result is a two's-complement 0/1 in the same layout.

/// 1 where x is negative (bit 31 set), else 0.
sfpi_inline sfpi::vInt _int_is_negative_(const sfpi::vInt x)
{
    return sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(x) >> 31);
}

/// 1 where x == 0, else 0. lz(x) is 32 only for an all-zero word, and 32 is the only leading-zero
/// count with bit 5 set.
sfpi_inline sfpi::vInt _int_is_zero_(const sfpi::vInt x)
{
    return sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(sfpi::lz(x)) >> 5);
}

/// 1 where x != 0, else 0: lz(x) - 32 is negative exactly when some bit of x is set.
sfpi_inline sfpi::vInt _int_is_nonzero_(const sfpi::vInt x)
{
    return _int_is_negative_(sfpi::as<sfpi::vInt>(sfpi::lz(x)) - 32);
}

/**
 * @brief Integer comparison of every element against zero: 1 where it holds, else 0.
 *
 * Exact over the whole int32 range, INT_MIN included: gtz/lez OR the v - 1 term with v itself,
 * so the wrap of INT_MIN - 1 to INT_MAX cannot reach the sign bit that decides.
 */
template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
sfpi_inline void _calculate_zero_comp_int_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        const sfpi::vInt v = sfpi::dst_reg[0];
        sfpi::vInt r;
        if constexpr (COMP_MODE == SfpuType::equal_zero)
        {
            r = _int_is_zero_(v);
        }
        else if constexpr (COMP_MODE == SfpuType::not_equal_zero)
        {
            r = _int_is_nonzero_(v);
        }
        else if constexpr (COMP_MODE == SfpuType::less_than_zero)
        {
            r = _int_is_negative_(v);
        }
        else if constexpr (COMP_MODE == SfpuType::greater_than_equal_zero)
        {
            r = _int_is_negative_(~v);
        }
        else if constexpr (COMP_MODE == SfpuType::greater_than_zero)
        {
            // v > 0  <=>  v >= 0 and v - 1 >= 0 (both signs clear).
            r = _int_is_negative_(~(v | (v - 1)));
        }
        else
        {
            static_assert(COMP_MODE == SfpuType::less_than_equal_zero, "not a comparison-to-zero mode");
            // v <= 0  <=>  v < 0 or v - 1 < 0. For v == INT_MIN the second term wraps to INT_MAX
            // and reads as false, but v's own sign already answers.
            r = _int_is_negative_(v | (v - 1));
        }
        sfpi::dst_reg[0] = r;
        sfpi::dst_reg++;
    }
}

/**
 * @brief v < s over ITERATIONS rows, for a scalar whose sign is SCALAR_NEGATIVE.
 *
 *   s >= 0:  v < s  <=>  v < 0, or v >= 0 and v - s < 0   =  sign(v | (v - s))
 *   s <  0:  v < s  <=>  v < 0 and v - s < 0              =  sign(v & (v - s))
 *
 * v - s is only consulted when v and s share a sign, where it cannot overflow; when the signs
 * differ, v's own sign decides and the OR (AND) makes it dominate. The scalar's sign is uniform,
 * so the split is a RISC branch rather than per-lane predication.
 */
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
            sfpi::dst_reg[0] = _int_is_negative_(v & diff);
        }
        else
        {
            sfpi::dst_reg[0] = _int_is_negative_(v | diff);
        }
        sfpi::dst_reg++;
    }
}

/**
 * @brief v > s over ITERATIONS rows: the mirror of _comp_unary_int_lt_rows_ on s - v and ~v.
 *
 *   s >= 0:  v > s  <=>  v >= 0 and s - v < 0             =  sign(~v & (s - v))
 *   s <  0:  v > s  <=>  v >= 0, or v < 0 and s - v < 0   =  sign(~v | (s - v))
 */
template <bool SCALAR_NEGATIVE, int ITERATIONS>
sfpi_inline void _comp_unary_int_gt_rows_(const sfpi::vInt s)
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        const sfpi::vInt v     = sfpi::dst_reg[0];
        const sfpi::vInt not_v = ~v;
        // s - v is formed after ~v so SFPIADD can overwrite v in place instead of copying it.
        const sfpi::vInt diff = s - v;
        if constexpr (SCALAR_NEGATIVE)
        {
            sfpi::dst_reg[0] = _int_is_negative_(not_v | diff);
        }
        else
        {
            sfpi::dst_reg[0] = _int_is_negative_(not_v & diff);
        }
        sfpi::dst_reg++;
    }
}

/// v > s (IS_GT) or v < s, splitting on the scalar's sign on the RISC.
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

/// Writes the integer 1 to every element: the answer to v <= INT_MAX and v >= INT_MIN.
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

/**
 * @brief Integer comparison of every element against a scalar: 1 where it holds, else 0.
 *
 * Exact over the whole int32 range for both operands. eq/ne are bitwise (XOR, then the zero
 * test); lt/gt split on the scalar's sign (see _comp_unary_int_lt_rows_); le/ge are lt/gt
 * against the neighbouring scalar, and the one scalar with no neighbour (INT_MAX for le, INT_MIN
 * for ge) is the comparison every int32 satisfies.
 */
template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
sfpi_inline void _calculate_comp_unary_int_(int scalar)
{
    if constexpr (COMP_MODE == SfpuType::unary_eq || COMP_MODE == SfpuType::unary_ne)
    {
        const sfpi::vInt s = scalar;
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            const sfpi::vInt v = sfpi::dst_reg[0];
            const sfpi::vInt x = v ^ s;
            if constexpr (COMP_MODE == SfpuType::unary_eq)
            {
                sfpi::dst_reg[0] = _int_is_zero_(x);
            }
            else
            {
                sfpi::dst_reg[0] = _int_is_nonzero_(x);
            }
            sfpi::dst_reg++;
        }
    }
    else if constexpr (COMP_MODE == SfpuType::unary_lt)
    {
        _comp_unary_int_ordered_<false, ITERATIONS>(scalar);
    }
    else if constexpr (COMP_MODE == SfpuType::unary_gt)
    {
        _comp_unary_int_ordered_<true, ITERATIONS>(scalar);
    }
    else if constexpr (COMP_MODE == SfpuType::unary_le)
    {
        // v <= s  <=>  v < s + 1
        if (scalar == std::numeric_limits<std::int32_t>::max())
        {
            _comp_unary_int_all_true_<ITERATIONS>();
        }
        else
        {
            _comp_unary_int_ordered_<false, ITERATIONS>(scalar + 1);
        }
    }
    else
    {
        static_assert(COMP_MODE == SfpuType::unary_ge, "not a unary integer comparison mode");
        // v >= s  <=>  v > s - 1
        if (scalar == std::numeric_limits<std::int32_t>::min())
        {
            _comp_unary_int_all_true_<ITERATIONS>();
        }
        else
        {
            _comp_unary_int_ordered_<true, ITERATIONS>(scalar - 1);
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
