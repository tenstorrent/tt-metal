// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_hardshrink_bf16_simple_algebraic.h"
#include "sfpi.h"

namespace ckernel::sfpu::bf16::hardshrink
{
// The rows load and store through the hold mode (ADDR_MOD_7, ADDR_MOD_3 past WH's SFPU base), which
// every SFPU init leaves at zero step, and an INCRWC after each replay steps DEST: ADDR_MOD_6 belongs
// to the FPU ops (matmul) a kernel may run between this init and its call. A threshold pair that folds
// its identity store (always on WH, at 11 slots on BH) records its own INCRWC.
template <typename Config, int Iterations = 8>
inline void calculate_simple_forward()
{
    static_assert(Iterations > 0 && Iterations % 2 == 0);
    {
        constexpr bool folds         = Config::kKind == 1 && Config::kBodySlots == 11;
        constexpr std::uint32_t step = folds ? 0u : 2u * Config::kRowsPerReplay;
        {
            constexpr std::uint32_t pin = Config::kBodySlots != 16 ? Config::kThresholdBits : (Config::kComparatorBf16 << 16) ^ 0x80000000u;
            TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, pin >> 16);
            TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, pin & 65535);
        }
        load_replay_buf<Exec>(
            0,
            Config::kBodySlots,
            [&]
            {
                {
                    ttpoly::hardshrink::simple_threshold_pair<ADDR_MOD_7, ADDR_MOD_7, Config::kBodySlots == 11>();
                }
                if constexpr (step)
                {
                    TTI_INCRWC(0, step, 0, 0);
                }
            });
#pragma GCC unroll 8
        for (int row = Config::kRowsPerReplay; row < Iterations; row += Config::kRowsPerReplay)
        {
            TTI_REPLAY(0, Config::kBodySlots, 0, 0);
            if constexpr (step)
            {
                TTI_INCRWC(0, step, 0, 0);
            }
        }
    }
}
} // namespace ckernel::sfpu::bf16::hardshrink
