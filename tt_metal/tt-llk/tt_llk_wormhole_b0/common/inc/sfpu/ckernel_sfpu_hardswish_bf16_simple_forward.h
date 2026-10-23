// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_hardswish_bf16_simple_algebraic.h"
#include "sfpi.h"

namespace ckernel::sfpu::bf16::hardswish
{
template <typename Config, int Iterations = 8>
inline void calculate_simple_forward()
{
    static_assert(Iterations > 0 && Iterations % 2 == 0);
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, Config::kSlopeBits >> 16);
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, Config::kSlopeBits & 65535);
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_USHORT, Config::kRawEqual);
    TTI_REPLAY(0, Config::kBodySlots, 1, 1);
    ttpoly::hardswish::simple_wh_gated_row<0, ADDR_MOD_3, ADDR_MOD_2>();
#pragma GCC unroll 8
    for (int row = Config::kRowsPerReplay; row < Iterations; row += Config::kRowsPerReplay)
    {
        TTI_REPLAY(0, Config::kBodySlots, 0, 0);
    }
}
} // namespace ckernel::sfpu::bf16::hardswish
