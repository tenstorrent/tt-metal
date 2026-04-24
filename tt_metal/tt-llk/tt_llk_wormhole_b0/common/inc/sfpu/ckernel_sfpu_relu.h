// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_addrmod.h"
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
inline void _calculate_lrelu_(std::uint32_t dst_index_in, std::uint32_t dst_index_out, const int iterations, std::uint32_t slope)
{
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
    TT_SFPLOADI(p_sfpu::LREG2, 10, slope & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, 8, slope >> 16);
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, dst_index_in * SFP_DST_TILE_ROWS); // load from dest into lreg[0]
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);                                         // condition - if value in LREG0 is negative //will set cc result reg
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0); // Multiply LREG0 * LREG2 (x * slope)
        TTI_SFPENCC(0, 0, 0, 0);                                                      // clear cc result reg
        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, dst_index_out * SFP_DST_TILE_ROWS); // store from lreg0 into dest register
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
inline void _relu_max_impl_(std::uint32_t dst_index_in, std::uint32_t dst_index_out, const int iterations, VecType threshold)
{
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
    for (int d = 0; d < iterations; d++)
    {
        VecType result = sfpi::dst_reg[dst_index_in * SFP_DST_TILE_ROWS];
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
        sfpi::dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = result;
        sfpi::dst_reg++;
    }
}

// Wrappers
template <typename VectorType, bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _relu_max_(std::uint32_t dst_index_in, std::uint32_t dst_index_out, T threshold)
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

    _relu_max_impl_<VectorType, APPROXIMATION_MODE, ITERATIONS>(dst_index_in, dst_index_out, ITERATIONS, v_threshold);
}

template <typename VecType, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_min_impl_(
    std::uint32_t dst_index_in, std::uint32_t dst_index_out, const int iterations, [[maybe_unused]] VecType threshold, int sfpload_instr_mod)
{
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
    for (int d = 0; d < iterations; d++)
    {
        // Load input tensor to lreg0
        TT_SFPLOAD(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_3, dst_index_in * SFP_DST_TILE_ROWS);
        // Copy value param from lreg2 to lreg1
        TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        // Swap and store maximum in lreg1, minimum in lreg0 (sign + magnitude format)
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, 1);
        // Store the result
        TT_SFPSTORE(p_sfpu::LREG1, sfpload_instr_mod, ADDR_MOD_3, dst_index_out * SFP_DST_TILE_ROWS);
        sfpi::dst_reg++;
    }
}

// Wrappers
template <typename VectorType, bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _relu_min_(std::uint32_t dst_index_in, std::uint32_t dst_index_out, T threshold)
{
    static_assert(std::is_same_v<VectorType, sfpi::vFloat> || std::is_same_v<VectorType, sfpi::vInt>, "VectorType must be sfpi::vFloat or sfpi::vInt");

    VectorType v_threshold;
    int scalar = threshold;
    if (scalar < 0)
    { // To convert from 2's complement to sign+magnitude
        scalar  = -scalar;
        int res = 0x80000000 | (scalar & 0x7FFFFFFF);
        scalar  = res;
    }
    int sfpload_instr_mod = DEFAULT;
    if constexpr (std::is_same_v<T, float>)
    {
        v_threshold = threshold;
    }
    else if constexpr (std::is_same_v<T, std::uint32_t>)
    {
        if constexpr (std::is_same_v<VectorType, sfpi::vInt>)
        {
            _sfpu_load_imm32_(p_sfpu::LREG2, scalar);
            sfpload_instr_mod = INT32_2S_COMP;
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

    _relu_min_impl_<VectorType, APPROXIMATION_MODE, ITERATIONS>(dst_index_in, dst_index_out, ITERATIONS, v_threshold, sfpload_instr_mod);
}

} // namespace sfpu
} // namespace ckernel
