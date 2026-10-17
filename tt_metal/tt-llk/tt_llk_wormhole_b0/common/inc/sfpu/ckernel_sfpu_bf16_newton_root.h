// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

// Sibling stock wrappers may include another generated config while this
// shared runtime is being parsed. Expose its adapter interface first.
namespace ckernel::sfpu::bf16
{
template <typename Config, int Iterations = 8>
inline void calculate_newton_root();
}

// rsqrt includes sqrt before declaring its callback; a composed sqrt config
// can therefore reach this runtime first. These declarations match the
// included stock definitions, without repeating their default arguments.
namespace ckernel::sfpu::bf16
{
// The selected WH sqrt row in 26 slots. TT-NN's datacopy delivers -0 and subnormals to DEST
// unflushed, so every raw word reaches the row, and the multiplies read a subnormal as a zero
// of its sign. The second step reads |x|, so -0 runs as +0, and halves its correction
// 1 - y*xy rather than xy: at x = +-0 or a positive subnormal xy is 0, whose halved exponent
// wraps to +Inf, while the correction is then 1. Powers of two scale exactly, so normal inputs
// keep the earlier row's words. Every zero or positive subnormal word thus ends at +0, and the
// raw compare 0x8000 - raw < 0 (sign set, rest nonzero) sends every negative word but -0 to NaN.
constexpr std::uint32_t kNewtonSqrtCompactSlots = 26u;

inline void newton_sqrt_wh_compact_row()
{
    TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_3, 0);                                  // x
    TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);                                // WH shifts its destination in place
    TTI_SFPSHFT(0xFFF, 0, p_sfpu::LREG0, 1);                                       // bits(x) >> 1
    TTI_SFPIADD(0, p_sfpu::LREG12, p_sfpu::LREG0, 6);                              // y = MAGIC - i
    TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);  // xy
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG3, 1);                                // -y
    TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);  // c = -y*xy
    TTI_SFPLOAD(p_sfpu::LREG7, sfpi::SFPLOAD_MOD0_FMT_UINT16, ADDR_MOD_3, 0);      // raw word
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG14, p_sfpu::LREG1, p_sfpu::LREG3, 0); // C2 + c
    TTI_SFPABS(0, p_sfpu::LREG2, p_sfpu::LREG2, 1);                                // |x|
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LREG13, p_sfpu::LREG1, 0);    // t = c*(C2+c) + C1
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_USHORT, 0x8000);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0); // y *= t
    TTI_SFPLOADI(p_sfpu::LREG6, 0, 0x7F80);                                       // +Inf
    TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0); // xy
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG3, 1);                               // -y
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG4, 0); // 1 - y*xy
    TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG6, 2);                              // |x| < +Inf
    TTI_SFPDIVP2(0xFFF, p_sfpu::LREG4, p_sfpu::LREG3, 1);                         // (1 - y*xy)/2
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::LREG0, 0);    // y = (1-y*xy)/2*xy + xy
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG7, 2); // 0x8000 - raw < 0: negative, not -0
    TTI_SFPLOADI(p_sfpu::LREG0, 0, 0x7FC0);
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LREG0,
                      sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B); // fp32 -> bf16 RNE
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_2, 0);           // dest += 2
}

template <typename Config, int Iterations>
inline void calculate_newton_root()
{
    static_assert(Iterations == 8 || Iterations == 32);
    static_assert(Config::kBodySlots == kNewtonSqrtCompactSlots);
    TTI_REPLAY(0, Config::kBodySlots, 1, 1);
    newton_sqrt_wh_compact_row();
#pragma GCC unroll 8
    for (int row = 1; row < Iterations; ++row)
    {
        TTI_REPLAY(0, Config::kBodySlots, 0, 0);
    }
}
} // namespace ckernel::sfpu::bf16
