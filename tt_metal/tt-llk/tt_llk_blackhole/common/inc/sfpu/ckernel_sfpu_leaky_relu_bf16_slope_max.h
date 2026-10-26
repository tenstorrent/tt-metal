// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu::bf16::leaky_relu
{
template <typename Config, int Iterations = 8>
inline void calculate_slope_max()
{
    static_assert(Iterations > 0 && Iterations % 2 == 0);
    // Preserve the selected S55 replay body. The upstream callback owns face
    // traversal and WH addrmod-base lifetime; no deployment tile reset belongs here.
    TTI_REPLAY(0, Config::kBodySlots, 1, 1);
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 2);
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);
    TTI_SFPMUL(p_sfpu::LREG12, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_SFPSETCC(0, p_sfpu::LREG1, 0, 0);
    TTI_SFPMUL(p_sfpu::LREG12, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
    TTI_SFPENCC(3, 0, 0, 10);
    // FP32 -> BF16 reads no VB; ttsim rejects a nonzero one, so VB is L0.
    TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    // ADDR_MOD_6 keeps stock's dest += 2, which other unary SFPU ops read, so each store advances.
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_6, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_6, 0);
#pragma GCC unroll 16
    for (int pair = 1; pair < Iterations / 2; ++pair)
    {
        TTI_REPLAY(0, Config::kBodySlots, 0, 0);
    }
}
} // namespace ckernel::sfpu::bf16::leaky_relu
