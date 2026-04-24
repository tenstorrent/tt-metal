// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_ops.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_square_(std::uint32_t dst_index_in, std::uint32_t dst_index_out)
{
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        // Load input from destination into LREG0
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, dst_index_in * SFP_DST_TILE_ROWS);
        // Multiply LREG0 * LREG0, store result in LREG0
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPNOP;
        // Store result back to destination
        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, dst_index_out * SFP_DST_TILE_ROWS);
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
