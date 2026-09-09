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
    // Pure sfpi: `v_if (v < 0) v *= slope` lowers to the same per-element
    // sfpload/sfpsetcc/sfpmul/sfpencc/sfpstore the raw path emitted (the raw code was
    // already the natural predicate-multiply pattern, with no fused condition-code or
    // SFPSWAP trick to lose), so the executed instruction stream is identical while the
    // sfpi backend records it into a replay buffer and shrinks the static code size.
    // This mirrors the Wormhole _calculate_lrelu_, which already ships this exact sfpi form.
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

// The layout DEST is accessed through. Only the integer datapath needs a non-default one:
// Dest holds int32 as sign+magnitude, and on Blackhole a bare vInt access to dst_reg
// defaults to DataLayout::I32, which does *no* conversion -- so the compare below would see
// the raw bits and a negative operand would lose against a positive one. DataLayout::SM32 is
// the converting layout here; it emits sfpi's software smag_to_int / int_to_smag around the
// access (Blackhole's INT32_2S_COMP load/store mode has no effect -- see the note in
// ckernel_sfpu_sub_int.h).
//
// Read the two layout names carefully, they are the opposite way round to the intuition:
// I32 is the raw one and SM32 is the converting one. On Wormhole the bare default for vInt is
// already SM32 (sfpi_funcs.h picks it per-arch), which is why only Blackhole was wrong.
//
// Default for vFloat, where .mode<Default>() is a no-op and SM32 would not even be a valid
// layout for the type.
template <typename VecType>
inline constexpr sfpi::DataLayout relu_dest_layout_v = std::is_same_v<VecType, sfpi::vInt> ? sfpi::DataLayout::SM32 : sfpi::DataLayout::Default;

// threshold_is_negative selects which of the two integer forms below runs. A plain runtime
// bool rather than a template parameter, because the threshold is a runtime scalar; it is
// uniform across lanes, so the branch sits outside the loop and costs nothing per element.
// Unused on the float path, which is why it defaults.
template <typename VecType, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_min_impl_(const int iterations, VecType threshold, const bool threshold_is_negative = false)
{
    constexpr sfpi::DataLayout LAYOUT = relu_dest_layout_v<VecType>;

    if constexpr (std::is_same_v<VecType, sfpi::vInt>)
    {
        // Once SM32 has converted DEST, `a` is a full-range two's-complement int32 -- and a
        // plain `a < threshold` is then not a safe compare. The SFPU signed compare subtracts
        // and tests the sign of the result, so it inverts its answer whenever the two operands
        // are >= 2^31 apart: at threshold -(2^31 - 1) every input >= 1 would be wrongly clamped
        // down to the threshold, and at a small positive threshold inputs near INT32_MIN would
        // wrongly pass through. Splitting on the sign of the threshold leaves only same-sign
        // operands to the subtracting compare, whose difference cannot overflow.
        //
        // Wormhole is immune without this: SFPSWAP orders its operands as sign+magnitude
        // instead of subtracting. metal's relu_clamp_int guards the same way on this arch
        // (hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_relu.h) and calls the
        // guard "NOT redundant" for exactly this reason.
        //
        // An INT_MIN threshold needs no saturating clamp here, unlike the hand-built
        // sign+magnitude encoding on Wormhole: DEST cannot represent -2^31 either, so no input
        // is ever below such a threshold, and with the guard above the clamp correctly never
        // fires. It is only the overflowing compare that made it look like it clamped
        // everything.
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
