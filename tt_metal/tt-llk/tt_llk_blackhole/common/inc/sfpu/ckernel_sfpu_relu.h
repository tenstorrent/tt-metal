// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_converter.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <typename T>
constexpr bool is_supported_relu_type_v = std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>;

template <bool APPROXIMATION_MODE>
inline void _calculate_lrelu_(const int iterations, std::uint32_t slope)
{
    const sfpi::vFloat slope_v = Converter::as_float(slope);
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];
        v_if (v < 0.0f)
        {
            v = v * slope_v;
        }
        v_endif;
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

sfpi_inline sfpi::vFloat _relu_max_body_(sfpi::vFloat val, sfpi::vFloat threshold)
{
    sfpi::vFloat result = val;
    v_if (result > threshold)
    {
        result = threshold;
    }
    v_endif;
    v_if (result < 0.0f)
    {
        result = 0.0f;
    }
    v_endif;
    return result;
}

template <typename VecType, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_max_impl_(const int iterations, VecType threshold)
{
    for (int d = 0; d < iterations; d++)
    {
        VecType result = sfpi::dst_reg[0];
        v_if (result > threshold)
        {
            result = threshold;
        }
        v_endif;
        v_if (result < 0)
        {
            result = 0;
        }
        v_endif;
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// Wrappers
template <typename VectorType, bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _relu_max_(T threshold)
{
    static_assert(std::is_same_v<VectorType, sfpi::vFloat> || std::is_same_v<VectorType, sfpi::vInt>, "VectorType must be sfpi::vFloat or sfpi::vInt");

    VectorType v_threshold;
    if constexpr (std::is_same_v<T, float>)
    {
        static_assert(
            std::is_same_v<VectorType, sfpi::vFloat>,
            "A float threshold requires VectorType == sfpi::vFloat: sfpi::vInt has no float constructor, so the assignment below would otherwise fail as an "
            "ambiguous conversion");
        v_threshold = threshold;
    }
    else if constexpr (std::is_same_v<T, std::uint32_t>)
    {
        float f = Converter::as_float(threshold);
        if constexpr (std::is_same_v<VectorType, sfpi::vInt>)
        {
            v_threshold = int(f);
        }
        else
        {
            v_threshold = f;
        }
    }
    else
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>, "Threshold type must be float or uint32_t");
    }

    _relu_max_impl_<VectorType, APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, v_threshold);
}

// The layout DEST is accessed through. Only the integer datapath needs a non-default one, and
// the two names are the opposite way round to the intuition: I32 is the raw one and SM32 is
// the converting one. Wormhole already defaults vInt to SM32, which is why only Blackhole was
// wrong.
template <typename VecType>
inline constexpr sfpi::DataLayout relu_dest_layout_v = std::is_same_v<VecType, sfpi::vInt> ? sfpi::DataLayout::SM32 : sfpi::DataLayout::Default;

// threshold_is_negative selects which of the two integer forms below runs; it is uniform
// across lanes, so the branch sits outside the loop. Unused on the float path, which is why
// it defaults.
template <typename VecType, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_min_impl_(const int iterations, VecType threshold, const bool threshold_is_negative = false)
{
    constexpr sfpi::DataLayout LAYOUT = relu_dest_layout_v<VecType>;

    if constexpr (std::is_same_v<VecType, sfpi::vInt>)
    {
        // A plain `a < threshold` is not a safe compare over the full int32 range: the signed
        // compare subtracts, so it inverts its answer once the operands are 2^31 or more apart.
        // Splitting on the sign of the threshold leaves only same-sign operands to it, whose
        // difference cannot overflow. Wormhole compares differently and needs no split.
        if (threshold_is_negative)
        {
            for (int d = 0; d < iterations; d++)
            {
                sfpi::vInt a = sfpi::dst_reg[0].mode<LAYOUT>();
                // a >= 0 > threshold keeps a, so only the negative lanes can lose, and there
                // both operands are negative.
                v_if (a < 0)
                {
                    v_if (a < threshold)
                    {
                        a = threshold;
                    }
                    v_endif;
                }
                v_endif;
                sfpi::dst_reg[0].mode<LAYOUT>() = a;
                sfpi::dst_reg++;
            }
        }
        else
        {
            for (int d = 0; d < iterations; d++)
            {
                sfpi::vInt a = sfpi::dst_reg[0].mode<LAYOUT>();
                // Lifting the negative lanes to a non-negative threshold first is the whole
                // answer for them, and it leaves only non-negative operands to the compare.
                v_if (a < 0)
                {
                    a = threshold;
                }
                v_endif;
                v_if (a < threshold)
                {
                    a = threshold;
                }
                v_endif;
                sfpi::dst_reg[0].mode<LAYOUT>() = a;
                sfpi::dst_reg++;
            }
        }
    }
    else
    {
        for (int d = 0; d < iterations; d++)
        {
            VecType a = sfpi::dst_reg[0].mode<LAYOUT>();
            v_if (a < threshold)
            {
                sfpi::dst_reg[0].mode<LAYOUT>() = threshold;
            }
            v_endif;
            sfpi::dst_reg++;
        }
    }
}

// Wrappers
template <typename VectorType, bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _relu_min_(T threshold)
{
    static_assert(std::is_same_v<VectorType, sfpi::vFloat> || std::is_same_v<VectorType, sfpi::vInt>, "VectorType must be sfpi::vFloat or sfpi::vInt");

    VectorType v_threshold;
    // Only the integer path reads this; the sign of the threshold picks the overflow-safe
    // compare in _relu_min_impl_.
    bool threshold_is_negative = false;
    if constexpr (std::is_same_v<T, float>)
    {
        static_assert(
            std::is_same_v<VectorType, sfpi::vFloat>,
            "A float threshold requires VectorType == sfpi::vFloat: sfpi::vInt has no float constructor, so the assignment below would otherwise fail as an "
            "ambiguous conversion");
        v_threshold = threshold;
    }
    else if constexpr (std::is_same_v<T, std::uint32_t>)
    {
        if constexpr (std::is_same_v<VectorType, sfpi::vInt>)
        {
            const int scalar      = static_cast<int>(threshold);
            v_threshold           = scalar;
            threshold_is_negative = scalar < 0;
        }
        else
        {
            v_threshold = Converter::as_float(threshold);
        }
    }
    else
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>, "Threshold type must be float or uint32_t");
    }

    _relu_min_impl_<VectorType, APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, v_threshold, threshold_is_negative);
}

} // namespace sfpu
} // namespace ckernel
