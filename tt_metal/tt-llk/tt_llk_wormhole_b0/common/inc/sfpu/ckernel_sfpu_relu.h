// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_converter.h"
#include "ckernel_sfpu_load_config.h"
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
        v_threshold = threshold;
    }
    else if constexpr (std::is_same_v<T, std::uint32_t>)
    {
        if constexpr (std::is_same_v<VectorType, sfpi::vInt>)
        {
            v_threshold = static_cast<int>(Converter::as_float(threshold));
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

    _relu_max_impl_<VectorType, APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, v_threshold);
}

// The threshold is read from LREG2, NOT from a parameter: every caller must have loaded it
// with _sfpu_load_imm32_ first. The dead `VecType threshold` argument this used to carry made
// the dependency look satisfied when it was not -- see _relu_min_ below and tt-llk#1120.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_min_impl_(const int iterations, InstrModLoadStore sfpload_instr_mod)
{
    for (int d = 0; d < iterations; d++)
    {
        // Load input tensor to lreg0
        TTI_SFPLOAD(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_3, 0);
        // Copy value param from lreg2 to lreg1
        TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        // Swap and store maximum in lreg1, minimum in lreg0 (sign + magnitude format)
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, 1);
        // Store the result
        TTI_SFPSTORE(p_sfpu::LREG1, sfpload_instr_mod, ADDR_MOD_3, 0);
        sfpi::dst_reg++;
    }
}

// Wrappers
template <typename VectorType, bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _relu_min_(T threshold)
{
    static_assert(std::is_same_v<VectorType, sfpi::vFloat> || std::is_same_v<VectorType, sfpi::vInt>, "VectorType must be sfpi::vFloat or sfpi::vInt");

    // _relu_min_impl_ takes the threshold from LREG2, so every branch below has to load it.
    // The T == float branch used to assign a local vector instead and leave LREG2 untouched,
    // so relu_min ran against whatever the previously executed SFPU kernel had left there --
    // order-dependent garbage, which is tt-llk#1120. Only the tt-llk test harness instantiates
    // T == float; the Compute API passes uint32_t, which is why no shipping op ever saw it.
    InstrModLoadStore sfpload_instr_mod = InstrModLoadStore::DEFAULT;
    if constexpr (std::is_same_v<T, float>)
    {
        static_assert(std::is_same_v<VectorType, sfpi::vFloat>, "A float threshold requires VectorType == sfpi::vFloat");
        _sfpu_load_imm32_(p_sfpu::LREG2, __builtin_bit_cast(std::uint32_t, threshold));
    }
    else if constexpr (std::is_same_v<T, std::uint32_t>)
    {
        if constexpr (std::is_same_v<VectorType, sfpi::vInt>)
        {
            // SFPSWAP orders sign+magnitude, so a 2's complement threshold is converted here.
            // Scoped to this branch: it is meaningless for a float threshold, where the old
            // unconditional `int scalar = threshold` merely truncated the value.
            int scalar = static_cast<int>(threshold);
            if (scalar < 0)
            {
                scalar  = -scalar;
                int res = 0x80000000 | (scalar & 0x7FFFFFFF);
                scalar  = res;
            }
            _sfpu_load_imm32_(p_sfpu::LREG2, scalar);
            sfpload_instr_mod = InstrModLoadStore::INT32_2S_COMP;
        }
        else
        {
            _sfpu_load_imm32_(p_sfpu::LREG2, threshold);
        }
    }
    else
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>, "Threshold type must be float or uint32_t");
    }

    _relu_min_impl_<APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, sfpload_instr_mod);
}

} // namespace sfpu
} // namespace ckernel
