// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// lse = scale * m + log(l) in fp32 from fp32 bit patterns, with integer arithmetic only (the data movement
// RISC-Vs have no FPU; soft-float costs ~70 cycles per operation, this ~5x less per row). Pure C++, no device
// headers, so the host can check it.
//
// Fixed point Q24 (value * 2^24) in int64: log(l) = e * ln2 + log(1 + i/128) from a table, plus linear
// interpolation on the remaining 16 mantissa bits (exact table entry for a bf16 l, error < 1e-5 otherwise);
// scale * m as the exact 24 x 24-bit mantissa product shifted into Q24 (truncation error 2^-24). The sum is
// rounded to the nearest fp32.

#pragma once

#include <stdint.h>

namespace sdpa_lse {

// -1e30f: the LSE of a row with no visible key. Finite, so a later max / exp merge gives a weight of 0
// instead of NaN.
constexpr uint32_t kNoKeyLseBits = 0xF149F2CAu;

// round(log(1 + i / 128) * 2^24), i = 0..128.
constexpr int32_t kLog1pQ24[129] = {
    0,        130563,   260117,   388679,   516263,   642884,   768556,   893295,   1017112,  1140023,  1262040,
    1383175,  1503443,  1622854,  1741421,  1859157,  1976071,  2092177,  2207485,  2322006,  2435750,  2548728,
    2660951,  2772428,  2883169,  2993184,  3102482,  3211073,  3318965,  3426168,  3532691,  3638541,  3743728,
    3848259,  3952143,  4055388,  4158001,  4259990,  4361364,  4462128,  4562291,  4661859,  4760840,  4859240,
    4957067,  5054326,  5151025,  5247170,  5342767,  5437822,  5532342,  5626332,  5719799,  5812748,  5905184,
    5997115,  6088544,  6179477,  6269921,  6359879,  6449358,  6538362,  6626896,  6714966,  6802576,  6889730,
    6976434,  7062693,  7148510,  7233890,  7318838,  7403359,  7487455,  7571132,  7654394,  7737245,  7819688,
    7901728,  7983370,  8064615,  8145469,  8225936,  8306018,  8385720,  8465045,  8543997,  8622579,  8700794,
    8778647,  8856140,  8933277,  9010061,  9086495,  9162582,  9238326,  9313729,  9388795,  9463527,  9537927,
    9611998,  9685745,  9759168,  9832271,  9905058,  9977530,  10049690, 10121541, 10193086, 10264327, 10335266,
    10405907, 10476252, 10546303, 10616063, 10685534, 10754719, 10823619, 10892237, 10960577, 11028638, 11096425,
    11163939, 11231183, 11298158, 11364866, 11431311, 11497493, 11563416, 11629080,
};
constexpr int64_t kLn2Q24 = 11629080;  // round(log(2) * 2^24)

// Nearest fp32 to q * 2^-24 (|q| < 2^63).
inline uint32_t q24_to_fp32_bits(int64_t q) {
    if (q == 0) {
        return 0;
    }
    const uint32_t sign = q < 0 ? 0x80000000u : 0u;
    uint64_t mag = q < 0 ? static_cast<uint64_t>(-q) : static_cast<uint64_t>(q);
    int32_t msb = 63 - __builtin_clzll(mag);
    uint32_t mant;
    if (msb > 23) {
        const int32_t drop = msb - 23;
        mant = static_cast<uint32_t>(mag >> drop);
        const uint64_t rem = mag & ((uint64_t{1} << drop) - 1);
        const uint64_t half = uint64_t{1} << (drop - 1);
        if (rem > half || (rem == half && (mant & 1u))) {
            if (++mant == (1u << 24)) {
                mant >>= 1;
                ++msb;
            }
        }
    } else {
        mant = static_cast<uint32_t>(mag) << (23 - msb);
    }
    // value = mant * 2^(msb - 23 - 24): biased exponent msb - 24 + 127, always normal for |q| < 2^63.
    return sign | (static_cast<uint32_t>(msb + 103) << 23) | (mant & 0x7FFFFFu);
}

// fp32 bits of scale * m + log(l); m_bits, l_bits, scale_bits are fp32 bit patterns (a bf16 value shifted left
// by 16 is one). kNoKeyLseBits when l is not a positive normal number or m is not finite (a row that saw no
// key), and also, out of the fixed-point range, when |scale * m| >= 2^38.
inline uint32_t lse_bits(uint32_t m_bits, uint32_t l_bits, uint32_t scale_bits) {
    const uint32_t l_exp = (l_bits >> 23) & 0xFFu;
    const uint32_t m_exp = (m_bits >> 23) & 0xFFu;
    if ((l_bits >> 31) != 0 || l_exp == 0 || l_exp == 0xFFu || m_exp == 0xFFu) {
        return kNoKeyLseBits;
    }
    // log(l) = (e - 127) * ln2 + log(1 + mantissa)
    const uint32_t idx = (l_bits >> 16) & 0x7Fu;
    const uint32_t frac = l_bits & 0xFFFFu;
    int64_t q = (static_cast<int64_t>(l_exp) - 127) * kLn2Q24 + kLog1pQ24[idx];
    if (frac != 0) {
        q += (static_cast<int64_t>(kLog1pQ24[idx + 1] - kLog1pQ24[idx]) * frac) >> 16;
    }
    // scale * m = mant_m * mant_s * 2^(e_m + e_s - 300); m = 0 or denormal counts as 0.
    const uint32_t s_exp = (scale_bits >> 23) & 0xFFu;
    if (m_exp != 0 && s_exp != 0) {
        const uint64_t product =
            static_cast<uint64_t>((m_bits & 0x7FFFFFu) | 0x800000u) * ((scale_bits & 0x7FFFFFu) | 0x800000u);
        const int32_t shift = static_cast<int32_t>(m_exp + s_exp) - 276;  // to Q24
        if (shift >= 14) {
            return kNoKeyLseBits;  // |scale * m| >= 2^38: never a real logit
        }
        int64_t sm = 0;
        if (shift >= 0) {
            sm = static_cast<int64_t>(product << shift);
        } else if (shift > -48) {
            sm = static_cast<int64_t>(product >> -shift);
        }
        q += ((m_bits ^ scale_bits) >> 31) != 0 ? -sm : sm;
    }
    return q24_to_fp32_bits(q);
}

}  // namespace sdpa_lse
