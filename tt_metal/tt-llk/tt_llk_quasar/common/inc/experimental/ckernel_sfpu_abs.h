// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// AI-generated — run_id: 2026-04-08_abs_quasar_2f52d870
#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{
// Calculates ABS for number of rows of output SFPU ops (Quasar = 2 rows)
inline void _calculate_abs_sfp_rows_(const int out_offset)
{
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 0); // load from dest into lreg[0]

    // Float absolute value (clears sign bit). Use the named constant from
    // sfpi_constants.h — SFPABS_MOD1_INT would do integer 2's-complement abs,
    // which is wrong for float data.
    TTI_SFPABS(p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPABS_MOD1_FLOAT);

    // Store result to output tile (out_offset = 0 when in-place)
    TT_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, out_offset);
}

// Implements element-wise absolute value: abs(x)
inline void _calculate_abs_(std::uint32_t dst_tile_index_in, std::uint32_t dst_tile_index_out, const int iterations)
{
    const int out_offset = (dst_tile_index_out - dst_tile_index_in) * 32;
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        _calculate_abs_sfp_rows_(out_offset);
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>(); // does the dest_reg++ (increments by 2 rows)
    }
}

} // namespace sfpu
} // namespace ckernel
