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

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_sign_(std::uint32_t dst_index_in, std::uint32_t dst_index_out, const int iterations, std::uint32_t exponent_size_8)
{
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
// All params are in FP16 format
// uint format = 1;
#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat v                                   = sfpi::dst_reg[dst_index_in * SFP_DST_TILE_ROWS];
        sfpi::dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = sfpi::vConst1;
        v_if (v < 0.0F)
        {
            sfpi::dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = sfpi::vConstNeg1;
        }
        v_endif;

        // param0 == 0 is Bfp8 format. It does not require bias removal.
        // param0 != 0 is Float16 format and exp bias needs to be removed for zero check.
        v_if (_sfpu_is_fp16_zero_(v, exponent_size_8))
        {
            sfpi::dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = sfpi::vConst0;
        }
        v_endif;

        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
