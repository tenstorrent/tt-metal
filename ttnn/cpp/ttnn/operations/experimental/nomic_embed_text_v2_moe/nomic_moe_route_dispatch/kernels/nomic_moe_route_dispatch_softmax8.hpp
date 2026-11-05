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
//  - m1 needs no reduction: x = exp(v - max) <= 1 with exp(0) == 1.0 exactly and rounding of x * rcp is monotone,
//    so max_lane(y) == rcp in every lane.
//  - m2 = max over the lanes whose y != m1 (one rotate butterfly).
//  - code = sum over lanes of 2^e + 2^16 (y == m1) or 2^(e+8) (y == m2), exact in fp32 (integers < 2^20); ties to
//    the lower expert fall out of ctz on the DM RISC (decode_top2). code >= 2^17 <=> two lanes tie at m1: w2 = m1.
//  - code + 2^23 keeps the integer in the mantissa of a normal float (no denormal at the packer).
// Equality tests are on the fp32 bits; the weights are y's bits.
//
// share_faces (G > 1) also bypasses _llk_math_eltwise_unary_sfpu_params_: no VectorMode runs face 0 or face 2 alone,
// and each group member computes only its own 16-token halves (Perf R3.1, perf_experiments/r3_softmax_split: 8x512
// whole op 29.5 -> 26.8 us, cmp_softmax end 15.5k -> 11.1k cycles, bit-identical).

#pragma once

#include <cstdint>

#ifdef TRISC_MATH
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace route_dispatch_softmax8 {
using namespace sfpi;

template <int N>
sfpi_inline vFloat ror(vFloat v) {
#pragma GCC unroll 8
    for (int step = 0; step < N; ++step) {
        v = subvec_shflror1(v);
    }
    return v;
}

// One 4-row band: sfpi dst_reg units are 2 DEST rows; +1 = odd columns, +32 = the next DEST tile (next contributor).
template <uint32_t G>
sfpi_inline void band() {
    vFloat even = dst_reg[0];
    vFloat odd = dst_reg[1];
    if constexpr (G >= 2) {
        even = even + dst_reg[32];
        odd = odd + dst_reg[33];
    }
    if constexpr (G == 3) {
        even = even + dst_reg[64];
        odd = odd + dst_reg[65];
    }
    // lanes 0..3: experts 0,2,4,6; lanes 4..7: experts 1,3,5,7
    vFloat v = even;
    const vFloat odd4 = ror<4>(odd);
    v_if((vConstTileId & 8) != 0) { v = odd4; }
    v_endif;

    vFloat m = sfpi::max(v, ror<4>(v));
    m = sfpi::max(m, ror<2>(m));
    m = sfpi::max(m, ror<1>(m));
    // log2(e) and -ln2_hi live in programmable constants (init)
    const vFloat x = ckernel::sfpu::_sfpu_exp_fp32_accurate_prgm_<false>(
        v - m, vConstFloatPrgm2, ckernel::sfpu::EXP_FP32_P0, ckernel::sfpu::EXP_FP32_P1);
    vFloat s = x + ror<4>(x);
    s = s + ror<2>(s);
    s = s + ror<1>(s);
    const vFloat rcp = ckernel::sfpu::sfpu_reciprocal_iter<2>(s);
    const vFloat y = x * rcp;

    vFloat y2 = y;
    v_if(as<vInt>(y) == as<vInt>(rcp)) { y2 = -1.0f; }
    v_endif;
    vFloat m2 = sfpi::max(y2, ror<4>(y2));
    m2 = sfpi::max(m2, ror<2>(m2));
    m2 = sfpi::max(m2, ror<1>(m2));

    // lane l = (TileId / 2) % 8 holds expert 2l (l < 4) or 2(l - 4) + 1
    const vInt lane2 = vConstTileId & 14;
    vInt expert = lane2;
    v_if(lane2 >= 8) { expert = lane2 - 7; }
    v_endif;
    vFloat code = 0.0f;
    v_if(as<vInt>(y) == as<vInt>(rcp)) { code = setexp(vFloat(1.0f), expert + 127) + 65536.0f; }
    v_elseif(as<vInt>(y2) == as<vInt>(m2)) { code = setexp(vFloat(1.0f), expert + 135); }
    v_endif;
    code = code + ror<4>(code);
    code = code + ror<2>(code);
    code = code + ror<1>(code);
    vFloat w2 = m2;
    v_if(code >= 131072.0f) { w2 = rcp; }
    v_endif;
    code = code + 8388608.0f;

    vFloat out = code;
    v_if(lane2 == 2) { out = rcp; }
    v_elseif(lane2 == 4) { out = w2; }
    v_endif;
    dst_reg[0] = out;
}

// One face (16 rows); VectorMode::C runs it on faces 0 and 2.
template <uint32_t G>
inline void face() {
#pragma GCC unroll 0
    for (int row_band = 0; row_band < 4; ++row_band) {
        band<G>();
        dst_reg += 2;
    }
}

// Split softmax: the faces of DEST tile 0 in this member's share (face 0: tokens 0..15, face 2: 16..31). Same
// sequence as VectorMode::C (face, 2 face increments, face), with the unowned face's call skipped.
template <uint32_t G>
inline void share_faces(bool face0, bool face2) {
    _llk_math_eltwise_sfpu_start_(0);
    if (face0) {
        face<G>();
    }
    _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    if (face2) {
        face<G>();
    }
    _llk_math_eltwise_sfpu_done_();
}

inline void init() {
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    ckernel::sfpu::sfpu_reciprocal_init<false>();  // vConstFloatPrgm0 = 2.0 for the Newton steps
    vConstFloatPrgm1 = ckernel::sfpu::EXP_FP32_LOG2E;
    vConstFloatPrgm2 = ckernel::sfpu::EXP_FP32_NEG_LN2_HI;
}
}  // namespace route_dispatch_softmax8
#endif
