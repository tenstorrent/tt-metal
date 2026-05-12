// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Option 4 — register-agnostic, fully-unrolled inner kernel for square.
//
// ITERATIONS is the only template arg (square has no op-specific knobs in
// the original). The kernel processes ITERATIONS pairs of rows starting at
// `input_offset` / `output_offset`. ITERATIONS is independent of slice
// geometry — for tiny tiles the wrapper may set it below ydim/2 to operate
// only on the populated rows.
//
// `input_offset` and `output_offset` are runtime ints expected to be
// constexpr at the call site; the wrapper supplies per-slice constexpr
// offsets so SFPLOAD / SFPSTORE encode the address as an immediate.

#pragma once

#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel
{
namespace sfpu
{

template <int ITERATIONS>
inline void _calculate_square_(const int input_offset, const int output_offset)
{
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d++)
    {
        TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, input_offset + (d << 1));
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TT_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, output_offset + (d << 1));
    }
}

} // namespace sfpu
} // namespace ckernel
