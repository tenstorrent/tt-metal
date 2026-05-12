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

inline void _calculate_square_(const int iterations, const int in_offset_idx, const int out_offset_idx)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        // Load from source tile into LREG0
        TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, in_offset_idx + (d << 1));
        // Multiply LREG0 * LREG0, store result in LREG0
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        // Store result back to destination tile
        TT_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, out_offset_idx + (d << 1));
    }
}

} // namespace sfpu
} // namespace ckernel
