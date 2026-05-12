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
// Functions for both Dest and SrcS registers
// ================================================

template <bool REG_32BIT_MODE, bool TARGET_SRCS>
inline void _calculate_square_(const int iterations)
{
    // 4 rows per slice of SrcS for 32bit mode, 8 for 16bit mode
    constexpr int slice_size          = REG_32BIT_MODE ? 4 : 8;
    constexpr int target_slice_offset = TARGET_SRCS ? 2 * slice_size : 0;

#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        // Load from first slice of SrcS into LREG0
        TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, (d << 1));
        // Multiply LREG0 * LREG0, store result in LREG0
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        // Store result back to third slice of SrcS
        TT_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, target_slice_offset + (d << 1));
    }
}

} // namespace sfpu
} // namespace ckernel
