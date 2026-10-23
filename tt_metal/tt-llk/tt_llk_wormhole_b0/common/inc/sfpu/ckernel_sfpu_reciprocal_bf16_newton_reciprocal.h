// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_reciprocal_bf16_replay_window.h"
#include "sfpi.h"

namespace ckernel::sfpu::bf16::reciprocal
{
// The mantissa_quadratic reciprocal (root_n = 1), two DEST rows per replay. Each row:
//   s = sign(x) 2^(254 - e) = (0x7F800000 - bits(x) with its mantissa cleared) * 0.5
//   n = -m, m = |x|'s mantissa in [1, 2)
//   y = (c0 n + c1) n + c2;  t = n y + 1;  y = y (t t + t) + y;  result = (y rounded to BF16) s.
// The seed c0, c1, c2 is whatever stock's reciprocal init loads into Prgm0..2, read and never
// written: recip_tile_init's round_to_bf16 seed or the shared one sfpu_reciprocal_init loads
// without it. With either, y (1 + t + t^2) rounds correctly for every BF16 mantissa (the
// fitter's WH FMA model over all 128), so the result does not depend on which init ran.
// The scale carries the sign, so a zero or subnormal exponent gives a signed Inf and an
// all-ones exponent (Inf, NaN) a signed zero, with no compare. Row A uses L0 (y), L2 (s) and
// L3 (x, then n, t and u); row B L4, L5 and L6; L7 holds 0x7F800000 for the call. Every FMA's
// result is read two slots after it issues.
constexpr std::uint32_t kNewtonReciprocalSlots = 26u;

inline void newton_reciprocal_pair_head()
{
    TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_3, 0);      // x (row A)
    TTI_SFPLOAD(p_sfpu::LREG6, 0, ADDR_MOD_3, 2);      // x (row B)
    TTI_SFPSETMAN(0, p_sfpu::LREG3, p_sfpu::LREG2, 1); // sign and exponent of x
    TTI_SFPSETMAN(0, p_sfpu::LREG6, p_sfpu::LREG5, 1);
    TTI_SFPSETMAN(0, p_sfpu::LCONST_neg1, p_sfpu::LREG3, 0); // n = -m
    TTI_SFPSETMAN(0, p_sfpu::LCONST_neg1, p_sfpu::LREG6, 0);
    TTI_SFPMAD(p_sfpu::LREG12, p_sfpu::LREG3, p_sfpu::LREG13, p_sfpu::LREG0, 0); // c0 n + c1
    TTI_SFPMAD(p_sfpu::LREG12, p_sfpu::LREG6, p_sfpu::LREG13, p_sfpu::LREG4, 0);
    TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG2, 6);                            // 0x7F800000 - bits
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG0, 0); // y = (..) n + c2
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG6, p_sfpu::LREG14, p_sfpu::LREG4, 0);
    TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG5, 6);
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG3, 0); // t = n y + 1
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG6, 0);
    TTI_SFPMULI(0x3f00, p_sfpu::LREG2, 0);                                     // s = sign(x) 2^(254 - e)
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG3, p_sfpu::LREG3, p_sfpu::LREG3, 0); // u = t t + t
}

inline void newton_reciprocal_pair_tail()
{
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG6, p_sfpu::LREG6, p_sfpu::LREG6, 0);
    TTI_SFPMULI(0x3f00, p_sfpu::LREG5, 0);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG0, p_sfpu::LREG0, 0); // y = y u + y
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG6, p_sfpu::LREG4, p_sfpu::LREG4, 0);
    // Round y to BF16 before the scale: s is a power of two, so y s is then exact and equals the
    // rounded product wherever that is normal. Rounding after the multiply instead flushed the
    // product just below 2^-126 at x = +-2^126, whose correctly rounded result is 2^-126.
    // FP32 -> BF16 reads no VB; ttsim rejects a nonzero one, so VB is L0.
    TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LREG0,
                      sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B); // fp32 -> bf16 RNE
    TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LREG4, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0); // y s
    TTI_SFPMUL(p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);
    // ADDR_MOD_3 (ADDR_MOD_7 past the SFPU's +4 base) holds DEST at zero step in every SFPU init;
    // ADDR_MOD_6 belongs to the FPU ops between calls, so an INCRWC after each pair steps DEST.
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);
    TTI_SFPSTORE(p_sfpu::LREG4, 0, ADDR_MOD_3, 2);
}

template <typename Config, int Iterations = 8>
inline void calculate_newton_reciprocal()
{
    static_assert(Iterations > 0 && Iterations % 2 == 0);
    static_assert(Config::kBodySlots == kNewtonReciprocalSlots);
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, 0x7F80); // +Inf bits
    auto row_head = [&] { newton_reciprocal_pair_head(); };
    auto row_tail = [&] { newton_reciprocal_pair_tail(); };
    record_split<0, Config::kBodySlots, 16>(row_head, row_tail);
    TTI_INCRWC(0, 4, 0, 0);
#pragma GCC unroll 8
    for (int pair = 1; pair < Iterations / 2; ++pair)
    {
        replay_split<0, Config::kBodySlots, 16>(row_head, row_tail);
        TTI_INCRWC(0, 4, 0, 0);
    }
}
} // namespace ckernel::sfpu::bf16::reciprocal
