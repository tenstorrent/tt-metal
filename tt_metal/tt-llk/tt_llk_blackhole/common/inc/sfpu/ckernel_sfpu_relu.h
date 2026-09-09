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

// Sign of an integer threshold. It is uniform across lanes, so it picks which of the integer
// forms below runs from outside the loop.
enum class ThresholdSign
{
    Negative,
    Zero,
    Positive
};

inline constexpr ThresholdSign threshold_sign_of(const int scalar)
{
    return scalar < 0 ? ThresholdSign::Negative : scalar == 0 ? ThresholdSign::Zero : ThresholdSign::Positive;
}

template <typename VecType, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_min_impl_(const int iterations, VecType threshold, const ThresholdSign threshold_sign)
{
    if constexpr (std::is_same_v<VecType, sfpi::vInt>)
    {
        // A plain `a < threshold` is not a safe compare over the full int32 range: the signed
        // compare subtracts, so it inverts its answer once the operands are 2^31 or more apart.
        // Splitting on the sign of the threshold leaves only same-sign operands to it, whose
        // difference cannot overflow. Wormhole compares differently and needs no split.
        if (threshold_sign == ThresholdSign::Negative)
        {
            for (int d = 0; d < iterations; d++)
            {
                sfpi::vInt a = sfpi::dst_reg[0];
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
                sfpi::dst_reg[0] = a;
                sfpi::dst_reg++;
            }
        }
        else if (threshold_sign == ThresholdSign::Zero)
        {
            // The sign test is the whole compare, which is the form relu_tile_int32 takes.
            for (int d = 0; d < iterations; d++)
            {
                sfpi::vInt a = sfpi::dst_reg[0];
                v_if (a < 0)
                {
                    a = threshold;
                }
                v_endif;
                sfpi::dst_reg[0] = a;
                sfpi::dst_reg++;
            }
        }
        else
        {
            for (int d = 0; d < iterations; d++)
            {
                sfpi::vInt a = sfpi::dst_reg[0];
                // Lifting the negative lanes to a positive threshold first is the whole answer
                // for them, and it leaves only non-negative operands to the compare.
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
                sfpi::dst_reg[0] = a;
                sfpi::dst_reg++;
            }
        }
    }
    else
    {
        for (int d = 0; d < iterations; d++)
        {
            VecType a = sfpi::dst_reg[0];
            v_if (a < threshold)
            {
                sfpi::dst_reg[0] = threshold;
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
    // The integer path alone reads this; its compare is the one that splits on the sign.
    ThresholdSign threshold_sign = ThresholdSign::Zero;
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
            const int scalar = static_cast<int>(threshold);
            v_threshold      = scalar;
            threshold_sign   = threshold_sign_of(scalar);
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

    _relu_min_impl_<VectorType, APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, v_threshold, threshold_sign);
}

} // namespace sfpu
} // namespace ckernel
