// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_softsign_bf16_abs_denominator_replay.h"
#include "sfpi.h"

namespace ckernel::sfpu::bf16::softsign
{
template <typename Config, int Iterations = 32>
inline void calculate_abs_denominator()
{
    static_assert(Iterations == 32, "selected compact traversal owns one whole tile");
    static_assert(Config::kBodySlots == 19);
    // Rows are addressed through ADDR_MOD_7 alone; the DEST counter steps one row after each.
#if defined(DISABLE_SFPLOADMACRO)
    // The same rows without the load macro or the replay buffer: identical results.
#pragma GCC unroll 1
    for (int row = 0; row < Iterations; ++row)
    {
        ttpoly::softsign::abs_denominator_bh_body<ADDR_MOD_7, ADDR_MOD_7, false, Config::kBoundBits>();
        TTI_INCRWC(0, 2, 0, 0);
    }
#else
    load_replay_buf<Exec>(
        0,
        Config::kBodySlots,
        [&]
        {
            ttpoly::softsign::abs_denominator_bh_body<ADDR_MOD_7, ADDR_MOD_7, true, Config::kBoundBits>();
            TTI_INCRWC(0, 2, 0, 0);
        });
#pragma GCC unroll 8
    for (int row = 1; row < Iterations; ++row)
    {
        TTI_REPLAY(0, Config::kBodySlots, 0, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
#endif
}
} // namespace ckernel::sfpu::bf16::softsign
