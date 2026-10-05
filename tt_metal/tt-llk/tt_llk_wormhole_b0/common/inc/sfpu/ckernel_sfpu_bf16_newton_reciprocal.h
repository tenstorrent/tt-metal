// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu::bf16 {
// The mantissa_quadratic reciprocal (root_n = 1), two DEST rows per replay. Each row:
//   s = sign(x) 2^(254 - e) = (0x7F800000 - bits(x) with its mantissa cleared) * 0.5
//   n = -m, m = |x|'s mantissa in [1, 2)
//   y = (c0 n + c1) n + c2;  t = n y + 1;  y = y (t t + t) + y;  result = y s, rounded to BF16.
// With the seed's 1.01% error e, y (1 + t + t^2) = (1 + e^3) / m rounds correctly for every
// mantissa; stock's one Newton step, y t + y, leaves two mantissas 0.5116 ULP off.
// The seed is the one stock's reciprocal init loads into Prgm0..2, read and never written.
// The scale carries the sign, so a zero or subnormal exponent gives a signed Inf and an
// all-ones exponent (Inf, NaN) a signed zero, with no compare. Row A uses L0 (y), L2 (s) and
// L3 (x, then n, t and u); row B L4, L5 and L6; L7 holds 0x7F800000 for the call. Every FMA's
// result is read two slots after it issues.
constexpr std::uint32_t kNewtonReciprocalSlots = 26u;

inline void newton_reciprocal_pair() {
    TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_3, 0);  // x (row A)
    TTI_SFPLOAD(p_sfpu::LREG6, 0, ADDR_MOD_3, 2);  // x (row B)
    TTI_SFPSETMAN(0, p_sfpu::LREG3, p_sfpu::LREG2, 1);  // sign and exponent of x
    TTI_SFPSETMAN(0, p_sfpu::LREG6, p_sfpu::LREG5, 1);
    TTI_SFPSETMAN(0, p_sfpu::LCONST_neg1, p_sfpu::LREG3, 0);  // n = -m
    TTI_SFPSETMAN(0, p_sfpu::LCONST_neg1, p_sfpu::LREG6, 0);
    TTI_SFPMAD(p_sfpu::LREG12, p_sfpu::LREG3, p_sfpu::LREG13, p_sfpu::LREG0, 0);  // c0 n + c1
    TTI_SFPMAD(p_sfpu::LREG12, p_sfpu::LREG6, p_sfpu::LREG13, p_sfpu::LREG4, 0);
    TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG2, 6);  // 0x7F800000 - bits
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG0, 0);  // y = (..) n + c2
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG6, p_sfpu::LREG14, p_sfpu::LREG4, 0);
    TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG5, 6);
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG3, 0);  // t = n y + 1
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG6, 0);
    TTI_SFPMULI(0x3f00, p_sfpu::LREG2, 0);  // s = sign(x) 2^(254 - e)
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG3, p_sfpu::LREG3, p_sfpu::LREG3, 0);  // u = t t + t
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG6, p_sfpu::LREG6, p_sfpu::LREG6, 0);
    TTI_SFPMULI(0x3f00, p_sfpu::LREG5, 0);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG0, p_sfpu::LREG0, 0);  // y = y u + y
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG6, p_sfpu::LREG4, p_sfpu::LREG4, 0);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);  // y s
    TTI_SFPMUL(p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);
    TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LREG0,
                      sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);  // fp32 -> bf16 RNE
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);
    TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, p_sfpu::LREG4, p_sfpu::LREG4, p_sfpu::LREG4,
                      sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPSTORE(p_sfpu::LREG4, 0, ADDR_MOD_2, 2);  // dest += 4
}

template <typename Config, int Iterations = 8>
inline void calculate_newton_reciprocal() {
    static_assert(Iterations > 0 && Iterations % 2 == 0);
    static_assert(Config::kBodySlots == kNewtonReciprocalSlots);
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, 0x7F80);  // +Inf bits
    TTI_REPLAY(0, Config::kBodySlots, 1, 1);
    newton_reciprocal_pair();
#pragma GCC unroll 8
    for (int pair = 1; pair < Iterations / 2; ++pair) {
        TTI_REPLAY(0, Config::kBodySlots, 0, 0);
    }
}
}  // namespace ckernel::sfpu::bf16
