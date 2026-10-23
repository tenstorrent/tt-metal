// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_hardswish_bf16_replay_window.h"
#include "ckernel_sfpu_hardswish_bf16_simple_algebraic.h"
#include "sfpi.h"

namespace ckernel::sfpu::bf16::hardswish
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
        constexpr bool folds         = Config::kKind == 1;
        constexpr std::uint32_t step = folds ? 0u : 2u * Config::kRowsPerReplay;
        TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, Config::kSlopeBits >> 16);
        TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, Config::kSlopeBits & 65535);
        TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_USHORT, Config::kRawEqual);
        // A row too long for the window records its first 16 slots; a threshold or paired row records whole.
        constexpr bool split               = Config::kKind != 1;
        constexpr std::uint32_t head_slots = split ? 16u : Config::kBodySlots;
        auto row_head                      = [&]() { ttpoly::hardswish::simple_wh_gated_row_head<0, ADDR_MOD_3, ADDR_MOD_3>(); };
        auto row_tail                      = [&]()
        {
            if constexpr (split)
            {
                ttpoly::hardswish::simple_wh_gated_row_tail<0, ADDR_MOD_3, ADDR_MOD_3>();
            }
        };
        record_split<0, Config::kBodySlots, head_slots>(row_head, row_tail);
        if constexpr (step)
        {
            TTI_INCRWC(0, step, 0, 0);
        }
#pragma GCC unroll 8
        for (int row = Config::kRowsPerReplay; row < Iterations; row += Config::kRowsPerReplay)
        {
            replay_split<0, Config::kBodySlots, head_slots>(row_head, row_tail);
            if constexpr (step)
            {
                TTI_INCRWC(0, step, 0, 0);
            }
        }
    }
}
} // namespace ckernel::sfpu::bf16::hardswish
