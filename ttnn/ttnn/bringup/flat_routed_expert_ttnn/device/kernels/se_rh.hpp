// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The h options (config routed_weight / fp8_act_quant) on the gate/up cores' fp32 DEST, SFPU (pack thread), between
// the activation and h's pack: DeepSeek-V4.1's routed expert w2(q(r * act(w1 x, w3 x))).
//   rh_wmul<H, W>   h tile H *= the row's routing weight (tile W holds row r's weight in every column of faces 0 / 2,
//                   se5_recv.cpp builds it from the routing_weights row): one fp32 SFPU multiply per value
//   rh_quant<H, S>  fp8 e4m3 quantize-dequantize of tile H per row of 32 values (one quantization block along the
//                   intermediate dim), the ue8m0 scale as DeepSeek's act_quant(h, 32, 'ue8m0'):
//                     amax = max |h| over the row, clamped below at 1e-4
//                     s    = 2^ceil(log2(fp32(amax * fp32(1/448))))      (on the exponent bits)
//                     h    = sign(h) * e4m3(min(|h| / s, 448)) * s       (RNE; |h| / s = |h| * 2^-k, exact)
//                   S is scratch (s and 1/s per row; the weight tile, whose values are no longer needed then)
//   rh_merge<A, B>  tile A's right half (faces 1 / 3) := tile B's (a split h tile's two halves, from two cores)
// The math is unified_routed_expert_ffn's fused_swiglu.cpp calculate_routed_h (P.2, bit-exact with the DeepSeek
// reference on exact inputs), split so each step addresses its own tiles.
// SFPU lanes: one load is 4 rows x 8 columns of a face (even or odd columns); each 8-lane sub-vector is one row.
// Tile t: dst_reg[32 t + 16 half + 2 g + {0 even, 1 odd} + {0 left face, 8 right face}] = rows 16 half + 4 g .. + 3.
#pragma once

#include <cstdint>

namespace ckernel::sfpu {

// |y| in [0, 448] -> its nearest e4m3fn magnitude (ties to even), fp32.
sfpi_inline sfpi::vFloat rh_e4m3_magnitude(sfpi::vFloat a) {
    // normals (|y| >= 2^-6): keep 3 mantissa bits, rounding on the integer bits
    sfpi::vUInt b = sfpi::as<sfpi::vUInt>(a);
    sfpi::vUInt lsb = (b >> 20) & 1;
    b = (b + lsb + sfpi::vUInt(0x7FFFF)) & sfpi::vUInt(0xFFF00000u);
    sfpi::vFloat q = sfpi::as<sfpi::vFloat>(b);
    // subnormals (|y| < 2^-6): multiples of 2^-9; adding 2^14 (ulp 2^-9) rounds to nearest even
    v_if(a < 0.015625f) { q = (a + 16384.0f) - 16384.0f; }
    v_endif;
    return q;
}

template <uint32_t H, uint32_t S, uint32_t OK>
sfpi_inline void rh_qdq_value(const uint32_t o) {
    constexpr uint32_t T = 32;
    const sfpi::vFloat inv = sfpi::dst_reg[S * T + o + 1];
    const sfpi::vFloat a = sfpi::min(sfpi::abs(sfpi::vFloat(sfpi::dst_reg[H * T + o + OK])) * inv, 448.0f);
    const sfpi::vFloat q = rh_e4m3_magnitude(a) * sfpi::vFloat(sfpi::dst_reg[S * T + o]);
    sfpi::dst_reg[H * T + o + OK] = sfpi::copysgn(q, sfpi::vFloat(sfpi::dst_reg[H * T + o + OK]));
}

template <uint32_t H, uint32_t W>
inline void rh_wmul() {
    constexpr uint32_t T = 32;
#pragma GCC unroll 0
    for (uint32_t half = 0; half < 2; ++half) {
#pragma GCC unroll 0
        for (uint32_t g = 0; g < 4; ++g) {
            const uint32_t o = half * 16 + g * 2;
            const sfpi::vFloat w = sfpi::dst_reg[W * T + o];
            sfpi::dst_reg[H * T + o] = sfpi::vFloat(sfpi::dst_reg[H * T + o]) * w;
            sfpi::dst_reg[H * T + o + 1] = sfpi::vFloat(sfpi::dst_reg[H * T + o + 1]) * w;
            sfpi::dst_reg[H * T + o + 8] = sfpi::vFloat(sfpi::dst_reg[H * T + o + 8]) * w;
            sfpi::dst_reg[H * T + o + 9] = sfpi::vFloat(sfpi::dst_reg[H * T + o + 9]) * w;
        }
    }
}

template <uint32_t H, uint32_t S>
inline void rh_quant() {
    constexpr uint32_t T = 32;
#pragma GCC unroll 0
    for (uint32_t half = 0; half < 2; ++half) {
#pragma GCC unroll 0
        for (uint32_t g = 0; g < 4; ++g) {
            const uint32_t o = half * 16 + g * 2;
            // row amax: max over the 4 column sets, then over the 8 lanes of each row (rotate 4, 2, 1)
            sfpi::vFloat m = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[H * T + o]));
            m = sfpi::max(m, sfpi::abs(sfpi::vFloat(sfpi::dst_reg[H * T + o + 1])));
            m = sfpi::max(m, sfpi::abs(sfpi::vFloat(sfpi::dst_reg[H * T + o + 8])));
            m = sfpi::max(m, sfpi::abs(sfpi::vFloat(sfpi::dst_reg[H * T + o + 9])));
            sfpi::vFloat r = sfpi::subvec_shflror1(m);
            r = sfpi::subvec_shflror1(r);
            r = sfpi::subvec_shflror1(r);
            r = sfpi::subvec_shflror1(r);
            m = sfpi::max(m, r);
            r = sfpi::subvec_shflror1(m);
            r = sfpi::subvec_shflror1(r);
            m = sfpi::max(m, r);
            r = sfpi::subvec_shflror1(m);
            m = sfpi::max(m, r);
            // ue8m0 scale s and 1/s from the exponent bits of amax / 448
            const sfpi::vFloat t = sfpi::max(m, 1e-4f) * 0.002232142724096775f;
            const sfpi::vUInt sb = (sfpi::as<sfpi::vUInt>(t) + sfpi::vUInt(0x7FFFFF)) & sfpi::vUInt(0x7F800000);
            sfpi::dst_reg[S * T + o] = sfpi::as<sfpi::vFloat>(sb);
            sfpi::dst_reg[S * T + o + 1] = sfpi::as<sfpi::vFloat>(sfpi::vUInt(0x7F000000) - sb);
            rh_qdq_value<H, S, 0>(o);
            rh_qdq_value<H, S, 1>(o);
            rh_qdq_value<H, S, 8>(o);
            rh_qdq_value<H, S, 9>(o);
        }
    }
}

template <uint32_t A, uint32_t B>
inline void rh_merge() {
    constexpr uint32_t T = 32;
#pragma GCC unroll 0
    for (uint32_t half = 0; half < 2; ++half) {
#pragma GCC unroll 0
        for (uint32_t g = 0; g < 4; ++g) {
            const uint32_t o = half * 16 + g * 2;
            sfpi::dst_reg[A * T + o + 8] = sfpi::vFloat(sfpi::dst_reg[B * T + o + 8]);
            sfpi::dst_reg[A * T + o + 9] = sfpi::vFloat(sfpi::dst_reg[B * T + o + 9]);
        }
    }
}

}  // namespace ckernel::sfpu
