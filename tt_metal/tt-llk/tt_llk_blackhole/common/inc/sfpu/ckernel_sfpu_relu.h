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
    // Branch-free clamp to [0, threshold]. min/max lower to SFPSWAP, so this replaces two
    // SFPSETCC/SFPENCC predicate blocks (11 slots/element on Blackhole) with two SFPSWAPs
    // (6 slots).
    //
    // Order matters: the predicated original clamped HIGH first and LOW second, so a
    // negative threshold yields 0, not the threshold. max(min(..)) reproduces that;
    // min(max(..)) does not -- which is also why sfpi::clamp() is not usable here: it is
    // defined as min(max(val, lower), upper), the order this kernel must not use.
    return sfpi::max(sfpi::min(val, threshold), 0.0f);
}

template <typename VecType, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_max_impl_(const int iterations, VecType threshold)
{
    for (int d = 0; d < iterations; d++)
    {
        VecType result = sfpi::dst_reg[0];
        if constexpr (std::is_same_v<VecType, sfpi::vFloat>)
        {
            // See _relu_max_body_: branch-free clamp, high bound applied first.
            result = sfpi::max(sfpi::min(result, threshold), 0.0f);
        }
        else
        {
            // sfpi::min/max cover vFloat and vSMag only on Wormhole/Blackhole: the vInt
            // overloads sit inside `#if __riscv_xtttensixqsr` in sfpi's include/sfpi_lib.h
            // -- min(vInt, int), max(vInt, int) and the vInt arm of min_max's enable_if
            // (checked against sfpi 7.71/7.72). That header ships with the toolchain and is
            // not checked in (tt_metal/tt-llk/tests/.gitignore), so grepping this repo for
            // the macro finds nothing. SFPSWAP also orders operands as sign+magnitude, so a
            // vInt clamp would need the 2's-complement conversion _relu_min_ does by hand.
            // Left predicated.
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
        }
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

template <typename VecType, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_min_impl_(const int iterations, VecType threshold)
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

// Wrappers
template <typename VectorType, bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _relu_min_(T threshold)
{
    static_assert(std::is_same_v<VectorType, sfpi::vFloat> || std::is_same_v<VectorType, sfpi::vInt>, "VectorType must be sfpi::vFloat or sfpi::vInt");

    VectorType v_threshold;
    if constexpr (std::is_same_v<T, float>)
    {
        v_threshold = threshold;
    }
    else if constexpr (std::is_same_v<T, std::uint32_t>)
    {
        if constexpr (std::is_same_v<VectorType, sfpi::vInt>)
        {
            v_threshold = static_cast<int>(threshold);
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

    _relu_min_impl_<VectorType, APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, v_threshold);
}

} // namespace sfpu
} // namespace ckernel
