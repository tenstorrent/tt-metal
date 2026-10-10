// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_relu.h"

namespace ckernel::sfpu::bf16
{
constexpr std::uint32_t kClampHold    = ADDR_MOD_3;
constexpr std::uint32_t kClampAdvance = ADDR_MOD_2;

template <typename Config>
inline void init_clamped_affine()
{
    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    // These ordinary registers must be pinned before recording, as in the
    // selected paired MAD body. They are not clamp operands (SWAP writes both).
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, Config::kSlopeBits >> 16);
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, Config::kSlopeBits & 0xffffu);
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, Config::kInterceptBits >> 16);
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, Config::kInterceptBits & 0xffffu);
}

template <typename Config, int Iterations>
inline void clamped_affine_pairs()
{
    static_assert(Iterations % 2 == 0);
    TTI_REPLAY(0, Config::kBodySlots, 1, 1);
    TTI_SFPLOAD(p_sfpu::LREG0, 0, kClampHold, 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, kClampHold, 2);
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG0, p_sfpu::LREG7, p_sfpu::LREG2, 0);
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG1, p_sfpu::LREG7, p_sfpu::LREG3, 0);
    TTI_SFPSWAP(0, p_sfpu::LCONST_0, p_sfpu::LREG2, 9);
    TTI_SFPSWAP(0, p_sfpu::LCONST_0, p_sfpu::LREG3, 9);
    TTI_SFPSWAP(0, p_sfpu::LCONST_1, p_sfpu::LREG2, 1);
    TTI_SFPNOP;
    TTI_SFPSWAP(0, p_sfpu::LCONST_1, p_sfpu::LREG3, 1);
    TTI_SFPNOP;
    TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, p_sfpu::LREG3, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    // ADDR_MOD_6 keeps stock's dest += 2, which other unary SFPU ops read, so each store advances.
    TTI_SFPSTORE(p_sfpu::LREG2, 0, kClampAdvance, 0);
    TTI_SFPSTORE(p_sfpu::LREG3, 0, kClampAdvance, 0);
#pragma GCC unroll 4
    for (int pair = 1; pair < Iterations / 2; ++pair)
    {
        TTI_REPLAY(0, Config::kBodySlots, 0, 0);
    }
}
} // namespace ckernel::sfpu::bf16
