// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_leaky_relu_bf16_replay_window.h"
#include "sfpi.h"

namespace ckernel::sfpu::bf16::leaky_relu
{
template <typename Config>
inline void slope_max_constants()
{
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, Config::kSlopeBits >> 16);
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, Config::kSlopeBits & 0xffffu);
}

template <typename Config, int Iterations = 8>
inline void calculate_slope_max()
{
    static_assert(Iterations > 0 && Iterations % 2 == 0);
    slope_max_constants<Config>();
    auto row_body = [&]()
    {
        TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
        TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 2);
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);
        TTI_SFPMUL(p_sfpu::LREG4, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, 0);
        TTI_SFPMUL(p_sfpu::LREG4, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
        TTI_SFPENCC(3, 0, 0, 10);
        // FP32 -> BF16 reads no VB; ttsim rejects a nonzero one, so VB is L0.
        TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
        TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 2);
    };
    record_row<0, Config::kBodySlots>(row_body);
    // DEST is addressed only through the hold mode, which every SFPU init leaves at zero and no FPU
    // op sets; the counter steps past the pair outside the record, so the replay keeps its length.
    TTI_INCRWC(0, 4, 0, 0);
#pragma GCC unroll 16
    for (int pair = 1; pair < Iterations / 2; ++pair)
    {
        replay_row<0, Config::kBodySlots>(row_body);
        TTI_INCRWC(0, 4, 0, 0);
    }
}
} // namespace ckernel::sfpu::bf16::leaky_relu
