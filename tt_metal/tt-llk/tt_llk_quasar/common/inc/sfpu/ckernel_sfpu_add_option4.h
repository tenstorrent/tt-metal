// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Option 4 — register-agnostic, fully-unrolled inner kernel for binary add.
//
// Offsets are ints supplied by the wrapper. They are constexpr per slot when
// the slot is SrcS and runtime when the slot is Dest (resolved from the
// per-operand dest tile index). Runtime offsets fall back to TT_SFPLOAD /
// TT_SFPSTORE register dest_reg_addr (replay-buffer cost).

#pragma once

#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel
{
namespace sfpu
{

template <int ITERATIONS>
inline void _calculate_add_(const int input0_offset, const int input1_offset, const int output_offset)
{
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d++)
    {
        TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, input0_offset + (d << 1));
        TT_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, input1_offset + (d << 1));
        TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, 0x0);
        TT_SFPSTORE(p_sfpu::LREG2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, output_offset + (d << 1));
    }
}

} // namespace sfpu
} // namespace ckernel
