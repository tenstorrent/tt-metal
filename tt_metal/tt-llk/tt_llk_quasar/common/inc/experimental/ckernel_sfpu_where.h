// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel
{
namespace sfpu
{

// API-parity no-op: Quasar does not program SFPLOADMACRO templates for where.
inline void _init_where_()
{
}

// Inner row processor — per-lane select: out = (cond == 0) ? false_val : true_val.
// Offsets are in SFPU dest_reg_addr units (rows * 2).
inline void _calculate_where_sfp_rows_(const int in0_offset_idx, const int in1_offset_idx, const int in2_offset_idx, const int out_offset_idx)
{
    TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, in0_offset_idx); // condition -> LREG0
    TT_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, in1_offset_idx); // true_val  -> LREG1

    TTI_SFPSETCC(0, p_sfpu::LREG0, 6); // CC := (LREG0 == 0) (mod1=6 = SFPSETCC_MOD1_LREG_EQ0)

    TT_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, in2_offset_idx); // false_val -> LREG1 only on CC-enabled lanes

    TTI_SFPENCC(0, 0); // reset CC: re-enable all lanes

    TT_SFPSTORE(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, out_offset_idx); // store selected value
}

// Outer iterations loop — mirrors Quasar multi-input _calculate_add_; offsets advance by d*2 per iteration.
inline void _calculate_where_(const int iterations, const int in0_offset_idx, const int in1_offset_idx, const int in2_offset_idx, const int out_offset_idx)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        _calculate_where_sfp_rows_(in0_offset_idx + (d << 1), in1_offset_idx + (d << 1), in2_offset_idx + (d << 1), out_offset_idx + (d << 1));
    }
}

} // namespace sfpu
} // namespace ckernel
