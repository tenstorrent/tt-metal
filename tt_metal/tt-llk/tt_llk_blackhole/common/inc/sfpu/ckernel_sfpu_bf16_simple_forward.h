// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace sfpi
{
#include "ckernel_sfpu_bf16_simple_algebraic.h"
}

namespace ckernel::sfpu::bf16
{
template <typename Config, int Iterations = 8>
inline void calculate_simple_forward()
{
    static_assert(Iterations > 0 && Iterations % 2 == 0);
    {
        constexpr std::uint32_t pin = Config::kBodySlots != 16 ? Config::kThresholdBits : (Config::kComparatorBf16 << 16) ^ 0x80000000u;
        TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, pin >> 16);
        TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, pin & 65535);
    }
    TTI_REPLAY(0, Config::kBodySlots, 1, 1);
    {
        sfpi::simple_threshold_pair<ADDR_MOD_7, ADDR_MOD_6, Config::kBodySlots == 11>();
    }
#pragma GCC unroll 8
    for (int row = Config::kRowsPerReplay; row < Iterations; row += Config::kRowsPerReplay)
    {
        TTI_REPLAY(0, Config::kBodySlots, 0, 0);
    }
}
} // namespace ckernel::sfpu::bf16
