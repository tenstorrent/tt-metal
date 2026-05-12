// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel
{
namespace sfpu
{

// ================================================
// Functions for Dest register
// ================================================
// Calculates SQUARE for number of rows of output SFPU ops (Quasar = 2 rows)
template <bool APPROXIMATION_MODE>
inline void _calculate_square_sfp_rows_()
{
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 0); // load from dest into lreg[0]
    // Multiply LREG0 * LREG0, store result in LREG0
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    // Store result back to destination
    TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 0);
}

inline void _calculate_square_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        _calculate_square_sfp_rows_<false>();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>(); // does the dest_reg++ (increments by 2 rows)
    }
}

// ================================================
// Functions for SrcS register
// ================================================
// Assumes dest base address is set correctly to the start of SrcS: 0x400
template <bool SRCS_32BIT_MODE>
inline void _calculate_square_srcs_(const int iterations)
{
    // 4 rows per slice in SrcS for 32bit mode, 8 for 16bit mode
    constexpr int slice_size = SRCS_32BIT_MODE ? 4 : 8;

#pragma GCC unroll 4
    for (int d = 0; d < iterations; d++)
    {
        // Load from first slice of SrcS into LREG0
        TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, (d << 1));
        // Multiply LREG0 * LREG0, store result in LREG0
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        // Store result back to third slice of SrcS
        TT_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 2 * slice_size + (d << 1));
    }
}

} // namespace sfpu
} // namespace ckernel
