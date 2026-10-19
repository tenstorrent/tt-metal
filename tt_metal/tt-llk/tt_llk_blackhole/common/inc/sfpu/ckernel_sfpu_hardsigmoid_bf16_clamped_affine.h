// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_relu.h"

namespace ckernel::sfpu::bf16::hardsigmoid
{
constexpr std::uint32_t kClampHold = ADDR_MOD_7;

template <typename Config>
inline void clamped_affine_terminal_pair()
{
    TTI_SFPLOAD(p_sfpu::LREG4, sfpi::SFPLOAD_MOD0_FMT_UINT16, kClampHold, 0);
    TTI_SFPLOAD(p_sfpu::LREG5, sfpi::SFPLOAD_MOD0_FMT_UINT16, kClampHold, 2);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, Config::kTerminalEqual);
    TTI_SFPXOR(0, p_sfpu::LREG0, p_sfpu::LREG4, 0);
    TTI_SFPXOR(0, p_sfpu::LREG0, p_sfpu::LREG5, 0);
    TTI_SFPAND(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LREG1, 1);
    TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TTI_SFPSETCC(0, p_sfpu::LREG4, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, Config::kTerminalOutput);
    TTI_SFPENCC(0, 0, 0, 0);
    TTI_SFPAND(p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LREG1, 1);
    TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TTI_SFPSETCC(0, p_sfpu::LREG5, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_FLOATB, Config::kTerminalOutput);
    TTI_SFPENCC(0, 0, 0, 0);
}

template <typename Config, int Iterations>
inline void clamped_affine_pairs()
{
    static_assert(Iterations % 2 == 0);
    // The paired MAD reads its slope and intercept from L6 and L7, loaded by the call itself: an
    // ordinary LReg holds nothing across other ops. SWAP writes both clamp operands, not these.
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, Config::kSlopeBits >> 16);
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, Config::kSlopeBits & 0xffffu);
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, Config::kInterceptBits >> 16);
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, Config::kInterceptBits & 0xffffu);
    load_replay_buf<Exec>(
        0,
        Config::kBodySlots,
        [&]
        {
            TTI_SFPLOAD(p_sfpu::LREG0, 0, kClampHold, 0);
            TTI_SFPLOAD(p_sfpu::LREG1, 0, kClampHold, 2);
            TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG0, p_sfpu::LREG7, p_sfpu::LREG2, 0);
            TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG1, p_sfpu::LREG7, p_sfpu::LREG3, 0);
            TTI_SFPSWAP(0, p_sfpu::LCONST_0, p_sfpu::LREG2, 9);
            TTI_SFPSWAP(0, p_sfpu::LCONST_0, p_sfpu::LREG3, 9);
            TTI_SFPSWAP(0, p_sfpu::LCONST_1, p_sfpu::LREG2, 1);
            TTI_SFPSWAP(0, p_sfpu::LCONST_1, p_sfpu::LREG3, 1);
            clamped_affine_terminal_pair<Config>();
            // FP32 -> BF16 reads no VB; ttsim rejects a nonzero one, so VB is L0.
            TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
            TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
            TTI_SFPSTORE(p_sfpu::LREG2, 0, kClampHold, 0);
            TTI_SFPSTORE(p_sfpu::LREG3, 0, kClampHold, 2);
            // DEST is addressed only through the hold mode, which every SFPU init leaves at zero and no FPU
            // op sets; the counter steps past the pair outside the record, so the replay keeps its length.
            TTI_INCRWC(0, 4, 0, 0);
        });
#pragma GCC unroll 4
    for (int pair = 1; pair < Iterations / 2; ++pair)
    {
        TTI_REPLAY(0, Config::kBodySlots, 0, 0);
        TTI_INCRWC(0, 4, 0, 0);
    }
}
} // namespace ckernel::sfpu::bf16::hardsigmoid
