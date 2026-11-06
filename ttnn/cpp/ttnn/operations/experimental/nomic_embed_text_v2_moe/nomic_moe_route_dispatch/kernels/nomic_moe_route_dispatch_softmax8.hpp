// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// logits_block + softmax_block + top-2 in ONE DEST window a tile, on the SFPU in exact fp32 (math thread only).
//
// Raw SFPI, bypassing eltwise_chain (sum + Where mask), reduce<MAX, REDUCE_ROW>, the sub + exp chain,
// reduce<SUM, REDUCE_ROW> + recip post-op and mul bcast-col. Perf 1 (perf_experiments/softmax_lean): 8x512, 4 tiles,
// G = 3: helper path 9475 cycles, this 5545; 1 tile G = 1: 2282 -> 1087. Capability gap: kernel_lib has no SFPU row
// reduction over a column range that broadcasts back across the row and no way to keep intermediates in DEST across
// a reduce, so the helper path round-trips four intermediate CBs and runs max / sum / mul on the FPU at tf32.
//
// Per 4-row band the 8 expert columns become one SFPU vector (even columns in lanes 0..3, odd ones rotated into
// lanes 4..7); row max and row sum are a 3-stage SFPSHFT2 rotate butterfly with the result in every lane. Columns
// >= 8 never enter a reduced lane, so the untrusted partial faces / W tile padding (NaN, Inf) never reach an expert.
// The partials sum in the fixed order (p0 + p1) + p2 on every member: bit-identical scores across a group.
// exp is the accurate fp32 exp in both approx modes (math_approx_mode = True gets the accurate result).
//
// Top-2 runs here too, while a band's 8 probabilities are one vector (Perf 2, perf_experiments/r2_sfpu_top2: 8x512
// softmax -> select end 12455 -> 9836 cycles with the per-tile publish, bit-identical slots / weights / counts; the
// RISC select drops from ~95 to ~47 cycles a token). Same capability gap: no kernel_lib per-row top-k over a lane
// range. Output a band: column 0 = code, column 2 = w1, column 4 = w2 (other columns: don't care).
//  - m1 needs no reduction: x = exp(v - max) <= 1 with exp(0) == 1.0 exactly, so the m1 lanes are those with x == 1
//    and w1 = rcp.
//  - m2 = rcp * (max of x over the other lanes; one rotate butterfly run alongside the sum's); rounding of x * rcp is
//    monotone, so a lane is at m2 iff y >= m2.
//  - code = sum over lanes of 2^e + 2^16 (m1 lane) or 2^(e+8) (m2 lane), exact in fp32 (integers < 2^20); ties to
//    the lower expert fall out of ctz on the DM RISC (decode_top2). code >= 2^17 <=> two lanes tie at m1: w2 = m1.
//  - code + 2^23 keeps the integer in the mantissa of a normal float (no denormal at the packer).
// Equality tests are on the fp32 bits; the weights are y's bits.
//
// face0_only (G > 1) also bypasses _llk_math_eltwise_unary_sfpu_params_: each group member computes only its own
// 16-token halves (Perf R3.1, perf_experiments/r3_softmax_split: 8x512 whole op 29.5 -> 26.8 us), and since Perf R4.2
// a pass copies only the half's face of each contributor into face 0 of DEST tiles 0 .. G-1, so no VectorMode fits.
//
// Band instruction diet (Perf R4.2, perf_experiments/r42_softmax_pass, from r3_softmax_diet): the max butterfly's
// first stage runs on the even / odd pair before the lane merge, exp is the same accurate fp32 sequence without its
// overflow arm (v - max <= 0), the sum and m2 butterflies interleave, the Newton reciprocal keeps -2 in an LREG, and
// the expert code constants come from a per-lane vConstFloatPrgm0. 8x512 pass 1204 -> 915 cycles together with the
// one-face copy / pack, bit-identical outputs on every perf row.

#pragma once

#include <cstdint>

#ifdef TRISC_MATH
#include "ckernel_sfpu_exp.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace route_dispatch_softmax8 {
using namespace sfpi;

sfpi_inline vFloat ror1(vFloat v) { return subvec_shflror1(v); }
sfpi_inline vFloat ror2(vFloat v) { return ror1(ror1(v)); }
sfpi_inline vFloat ror4(vFloat v) { return ror2(ror2(v)); }

// p0 [+ p1 [+ p2]] of one column parity (sfpi dst_reg units are 2 DEST rows; +1 = odd columns, +32 = next tile).
template <uint32_t G, int off>
sfpi_inline vFloat contributors(const int col) {
    vFloat v = dst_reg[off + col];
    if constexpr (G >= 2) {
        v = v + dst_reg[off + col + 32];
    }
    if constexpr (G == 3) {
        v = v + dst_reg[off + col + 64];
    }
    return v;
}

// One 4-row band at DEST offset `off`. neg2 = -2.0 and c4 (the exp's last Horner coefficient) stay in LREGs.
template <uint32_t G, int off>
sfpi_inline void band(const vFloat neg2, const vFloat c4) {
    const vFloat odd = contributors<G, off>(1);
    const vFloat even = contributors<G, off>(0);
    // lanes 0..3: experts 0,2,4,6; lanes 4..7: experts 1,3,5,7. a = the max butterfly's ror4 stage in lanes 0..3.
    vFloat a = sfpi::max(even, odd);
    const vFloat odd4 = ror4(odd);
    const vFloat a4 = ror4(a);
    vFloat v = even;
    v_if(vConstFloatPrgm0 < 0.0f) {
        v = odd4;
        a = a4;
    }
    v_endif;
    vFloat m = sfpi::max(a, ror2(a));
    m = sfpi::max(m, ror1(m));

    // x = exp(v - m): the accurate fp32 exp (log2(e), -ln2_hi in vConstFloatPrgm1 / 2) without the overflow arm
    vFloat x;
    vInt e;
    {
        const vFloat arg = v - m;
        vInt i;
        vFloat f, r, j, y;
        vSMag16 sm;
        j = vConstFloatPrgm1 * arg;
        r = ckernel::sfpu::EXP_FP32_P0;
        sm = convert<vSMag16>(j, RoundMode::Nearest);
        j = convert<vFloat>(sm, RoundMode::Nearest);
        f = j * vConstFloatPrgm2 + arg;
        f = j * -1.42860677e-6f + f;
        r = r * f + ckernel::sfpu::EXP_FP32_P1;
        r = r * f + 4.16695364e-2f;
        r = r * f + 1.66664720e-1f;
        r = r * f + c4;
        i = abs(as<vInt>(sm));
        y = r * f + 1.0f;
        i = as<vInt>(copysgn(as<vFloat>(i), j));
        r = y * f + 1.0f;
        e = exexp(r, ExponentMode::Biased) + i;
        x = setexp(r, e);
        v_if(nearby(e < 1)) { x = 0.0f; }
        v_endif;
    }
    vFloat x2 = x;
    v_if(e >= 127) { x2 = -1.0f; }
    v_endif;

    // sum (the fixed ror4, ror2, ror1 order) and the mx2 max
    vFloat s = x + ror4(x);
    vFloat mx = sfpi::max(x2, ror4(x2));
    s = s + ror2(s);
    mx = sfpi::max(mx, ror2(mx));
    s = s + ror1(s);
    mx = sfpi::max(mx, ror1(mx));

    vFloat rcp = approx_recip(s);
    vFloat t = s * rcp + neg2;
    rcp = rcp * -t;
    t = s * rcp + neg2;
    rcp = rcp * -t;
    const vFloat y = x * rcp;
    const vFloat m2 = mx * rcp;

    vFloat code = 0.0f;
    v_if(as<vInt>(y) >= as<vInt>(m2)) { code = (abs(vConstFloatPrgm0) - 65536.0f) * 256.0f; }
    v_endif;
    v_if(x2 < 0.0f) { code = abs(vConstFloatPrgm0); }
    v_endif;
    code = code + ror4(code);
    code = code + ror2(code);
    code = code + ror1(code);
    // even columns: code; odd columns: w1 (lanes 0..3, column 1) and w2 (lanes 4..7, column 9)
    vFloat w2 = m2;
    v_if(code >= 131072.0f) { w2 = rcp; }
    v_endif;
    const vInt lane2 = vConstTileId & 14;
    vFloat out = code + 8388608.0f;
    v_if(lane2 == 2) { out = rcp; }
    v_elseif(lane2 == 4) { out = w2; }
    v_endif;
    dst_reg[off] = out;
}

// One face (16 rows); G = 1: VectorMode::C runs it on faces 0 and 2.
template <uint32_t G>
inline void face() {
    const vFloat neg2 = -2.0f;
    const vFloat c4 = 4.99999851e-1f;
#pragma GCC unroll 0
    for (int row_band = 0; row_band < 4; ++row_band) {
        band<G, 0>(neg2, c4);
        dst_reg += 2;
    }
}

// Face-IO split pass: the contributors' face sits in face 0 of DEST tiles 0 .. G-1. `hook` runs on the MATH RISC
// after `split` of the 4 row bands are issued (the SFPU keeps draining them): the next pass's dest address goes out
// there (Perf R4.3).
template <uint32_t G, uint32_t split, typename Hook>
inline void face0_only(Hook hook) {
    static_assert(split <= 4, "a face has 4 row bands");
    _llk_math_eltwise_sfpu_start_(0);
    const vFloat neg2 = -2.0f;
    const vFloat c4 = 4.99999851e-1f;
#pragma GCC unroll 0
    for (int row_band = 0; row_band < (int)split; ++row_band) {
        band<G, 0>(neg2, c4);
        dst_reg += 2;
    }
    hook();
#pragma GCC unroll 0
    for (int row_band = split; row_band < 4; ++row_band) {
        band<G, 0>(neg2, c4);
        dst_reg += 2;
    }
    _llk_math_eltwise_sfpu_done_();
}

inline void init() {
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    // per lane: +-(2^expert + 2^16), negative on the odd-expert lanes 4..7; lane l = (TileId / 2) % 8 holds expert 2l
    // (l < 4) or 2(l - 4) + 1
    const vInt lane2 = vConstTileId & 14;
    vInt expert = lane2;
    v_if(lane2 >= 8) { expert = lane2 - 7; }
    v_endif;
    vFloat q = setexp(vFloat(1.0f), expert + 127) + 65536.0f;
    v_if(lane2 >= 8) { q = -q; }
    v_endif;
    vConstFloatPrgm0 = q;
    vConstFloatPrgm1 = ckernel::sfpu::EXP_FP32_LOG2E;
    vConstFloatPrgm2 = ckernel::sfpu::EXP_FP32_NEG_LN2_HI;
}
}  // namespace route_dispatch_softmax8
#endif
