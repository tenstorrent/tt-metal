// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post — the hc step (DeepSeek-V4 hc_split_sinkhorn after the projection) as one custom fp32
// SFPU program over "hc lane" tiles, plus its eltwise_chain element (HcProgram).
//
// Lane layouts (op_design.md → Dataflow "stats → hc lanes"); quantity k = 0..23 are the stats mix
// columns, k = 24 the sum of squares. Every op below is lane-wise between quantities.
//  * ROBUST (any arch): quantity k occupies dest rows 4·(k%16)..+3 of lane tile k/16, all 16
//    columns — SFPU vectors dst_reg[2k] and dst_reg[2k+1]. Token t sits at rel row t/16, col t%16;
//    rel rows 2..3 duplicate rows 0..1. Each quantity is a whole number of SFPU loads, so it is
//    correct whatever the intra-load datum mapping; both halves are processed.
//  * DENSE (Blackhole, where the mapping was probed on device: one SFPU load = 4 dest rows x the 8
//    even or the 8 odd columns; processing only vector 2k of the robust layout left every odd
//    token untouched): quantity k is ONE load, dst_reg[k] — dest rows 4·(k/2)..+3, columns of
//    parity k%2. Token t at rel row t/8, col 2·(t%8) + k%2. All 25 quantities fit one lane tile,
//    the program runs once, and the writer moves half the words.
//
// Program per token (all fp32, lane-wise, exact transcendentals, no FPU operand):
//   r     = rsqrt(ss * (1/width) + norm_eps)                    (magic seed + 3 Newton steps)
//   pre   = sigmoid(a_pre  * (mix * r) + b) + hc_eps            (k = 0..n-1)
//   post  = 2 sigmoid(a_post * (mix * r) + b)                   (k = n..2n-1)
//   L     = a_res * (mix * r) + b, n x n row-major              (k = 2n..)
//   M     = softmax_rows(L) (row max subtracted) + hc_eps
//   M     = M / (col sums + hc_eps)
//   repeat iters - 1:  M = M / (row sums + hc_eps);  M = M / (col sums + hc_eps)
// Divisions are an exact reciprocal (approx + 2 Newton) times a multiply. Results overwrite the
// quantity slots 0..n(n+2)-1 in place; slot n(n+2) (ss) ends up holding r (don't-care).

#pragma once

#include <cstdint>

// Compute-only parts (the chain element and the SFPU program) are compiled for the TRISCs only;
// the writer includes this header for the lane layout constants.
#if defined(TRISC_MATH) || defined(TRISC_PACK) || defined(TRISC_UNPACK)
#define MHC_POST_HC_COMPUTE 1
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#endif

#ifdef TRISC_MATH
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/ckernel_sfpu_converter.h"
#endif

namespace mhc_post_hc {

constexpr uint32_t N = 4;
constexpr uint32_t NG = N * (N + 2);  // 24 hc quantities
constexpr uint32_t SS_Q = NG;         // quantity index of the sum of squares
#ifdef ARCH_BLACKHOLE
constexpr bool DENSE = true;  // the layout knob (see above)
#else
constexpr bool DENSE = false;
#endif
constexpr uint32_t LANE_TILES = DENSE ? 1 : 2;              // tiles the hc program works on (host CB holds 2)
constexpr uint32_t HALVES = DENSE ? 1 : 2;                  // SFPU passes over the quantities
constexpr uint32_t VEC_STRIDE = DENSE ? 1 : 2;              // SFPU vectors per quantity step
constexpr uint32_t LANE_CB_PAGES = 2;                       // cb_hc_lanes / cb_hc_out capacity (host HC_LANE_TILES)
constexpr uint32_t LANE_SETS = LANE_CB_PAGES / LANE_TILES;  // hc rows in flight writer <-> compute

// Runtime parameter block (compute RT args, fp32 bits unless noted).
constexpr uint32_t P_BASE = 0;            // b_0 .. b_23
constexpr uint32_t P_A_PRE = NG + 0;      // a_pre
constexpr uint32_t P_A_POST = NG + 1;     // a_post
constexpr uint32_t P_A_RES = NG + 2;      // a_res
constexpr uint32_t P_INV_WIDTH = NG + 3;  // 1 / hc_norm_width (host double -> fp32)
constexpr uint32_t P_NORM_EPS = NG + 4;   // hc_norm_eps
constexpr uint32_t P_HC_EPS = NG + 5;     // hc_eps
constexpr uint32_t P_ITERS = NG + 6;      // hc_sinkhorn_iters (integer)
constexpr uint32_t NUM_PARAMS = NG + 7;

// Writer-side lane addressing: fp32 word of (quantity k, token t), counted from lane tile 0 (a
// tile is 1024 words; faces stacked, so dest row R, column c is word 16·R + c). ROBUST layout: the
// duplicate copy sits at + LANE_DUP.
constexpr uint32_t lane_word(uint32_t k, uint32_t t) {
    return DENSE ? 64 * (k / 2) + 16 * (t / 8) + 2 * (t % 8) + (k % 2) : (k / 16) * 1024 + (k % 16) * 64 + t;
}
constexpr uint32_t LANE_DUP = 32;

// NATURAL layout (Perf 1, Blackhole): the writer hands compute the stats tile verbatim (one local
// 4 KB NoC copy); compute copies it to DEST and transposes every 16x16 face in place (32-bit
// transpose_dest, faces stay put: face (tokens a, quantities b) becomes (quantities b, tokens a)),
// regroups each 4-quantity row group with SFPLOAD x4 + SFPTRANSP + SFPSTORE x4 so every quantity is
// ONE SFPU vector (all 32 tokens) at nat_addr(k), runs the hc program on those vectors, undoes the
// regroup, zeroes quantities 24..31 and face-transposes back, so the packed tile IS the hc page (row
// token, col quantity). Replaces the writer's two 800-word RISC permutes (stats -> lanes, lanes ->
// hc page: 6.7 + 5.9 us on the root core's critical tail at the focus shape; focus collapse
// 107.6 -> 96.9 us in the bake-off, bit-identical). Raw LLK (SFPLOAD / SFPTRANSP / SFPSTORE,
// transpose_dest) bypasses the eltwise_chain HcProgram element: no helper exposes an in-DEST
// row -> lane regroup around an SFPU body. Carve-out: the regroup addresses are written for the
// Blackhole SFPU lane mapping (DENSE); other arches keep the writer permutes (ROBUST layout).
constexpr bool NATURAL = DENSE;
// SFPLOAD address (dest rows, +2 = odd columns) of quantity k's vector after the regroup. After the
// per-face transpose, quantity rows 0..15 live in faces 0 (tokens 0..15) and 2 (tokens 16..31),
// rows 16..31 in faces 1 and 3. Row group m = k/4: face rows 4(m%4)..+3 of face (m/4) and face
// (m/4) + 2; lane block k%4 of {f even, f odd, f+2 even, f+2 odd}.
constexpr uint32_t nat_base(uint32_t m) { return (m / 4) * 16 + (m % 4) * 4; }
constexpr uint32_t nat_off(uint32_t i) { return i == 0 ? 0 : i == 1 ? 2 : i == 2 ? 32 : 34; }
constexpr uint32_t nat_addr(uint32_t k) { return nat_base(k / 4) + nat_off(k % 4); }

}  // namespace mhc_post_hc

#ifdef TRISC_MATH
namespace ckernel::sfpu {

namespace mhc_hc_detail {

// SFPU vector index of quantity k (this half)
constexpr uint32_t q(uint32_t k) {
    return mhc_post_hc::NATURAL ? mhc_post_hc::nat_addr(k) / 2 : mhc_post_hc::VEC_STRIDE * k;
}

sfpi_inline sfpi::vFloat bits(uint32_t v) { return Converter::as_float(v); }

sfpi_inline sfpi::vFloat ld(uint32_t idx) { return sfpi::dst_reg[idx]; }

// 1 / x for finite x > 0: hardware approximation + 2 Newton steps (vConstFloatPrgm0 = 2.0).
sfpi_inline sfpi::vFloat recip(sfpi::vFloat x) { return sfpu_reciprocal_iter<2>(x); }

// 1 / sqrt(x) for finite x > 0: bit-trick seed + 3 Newton steps (~3e-11 before fp32 rounding).
sfpi_inline sfpi::vFloat rsqrt(sfpi::vFloat x) {
    sfpi::vInt i = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(x) >> 1);
    sfpi::vFloat y = sfpi::as<sfpi::vFloat>(sfpi::vInt(0x5f3759df) - i);
    sfpi::vFloat hx = x * 0.5f;
#pragma GCC unroll 3
    for (int it = 0; it < 3; ++it) {
        y = y * (1.5f - hx * y * y);
    }
    return y;
}

// sigmoid(t) = 1 / (1 + exp(-t)), exact exp.
sfpi_inline sfpi::vFloat sigmoid(sfpi::vFloat t) {
    sfpi::vFloat e = _sfpu_exp_fp32_accurate_<false>(-t);
    return recip(e + 1.0f);
}

// M[j][k] at quantity 2n + j*n + k.
constexpr uint32_t m(uint32_t j, uint32_t k) { return q(2 * mhc_post_hc::N + j * mhc_post_hc::N + k); }

sfpi_inline void row_normalize(uint32_t hc_eps_bits) {
    constexpr uint32_t n = mhc_post_hc::N;
#pragma GCC unroll 4
    for (uint32_t j = 0; j < n; ++j) {
        sfpi::vFloat s = ld(m(j, 0)) + ld(m(j, 1));
        s = s + sfpi::dst_reg[m(j, 2)];
        s = s + sfpi::dst_reg[m(j, 3)];
        sfpi::vFloat inv = recip(s + bits(hc_eps_bits));
#pragma GCC unroll 4
        for (uint32_t k = 0; k < n; ++k) {
            sfpi::dst_reg[m(j, k)] = sfpi::dst_reg[m(j, k)] * inv;
        }
    }
}

sfpi_inline void col_normalize(uint32_t hc_eps_bits) {
    constexpr uint32_t n = mhc_post_hc::N;
#pragma GCC unroll 4
    for (uint32_t k = 0; k < n; ++k) {
        sfpi::vFloat s = ld(m(0, k)) + ld(m(1, k));
        s = s + sfpi::dst_reg[m(2, k)];
        s = s + sfpi::dst_reg[m(3, k)];
        sfpi::vFloat inv = recip(s + bits(hc_eps_bits));
#pragma GCC unroll 4
        for (uint32_t j = 0; j < n; ++j) {
            sfpi::dst_reg[m(j, k)] = sfpi::dst_reg[m(j, k)] * inv;
        }
    }
}

// ---- Sinkhorn in reciprocal-scale form (Perf 1, NATURAL / Blackhole) ----
// M = diag(u) K diag(v) with K = softmax + eps read-only in the M slots. Track P_j = 1/u_j and
// Q_k = 1/v_k:  row step P_j <- (K v)_j + eps * P_j, u_j = 1/P_j  (== u/(u s + eps));
//               col step Q_k <- (K^T u)_k + eps * Q_k, v_k = 1/Q_k.
// One MAD + one reciprocal per target instead of 4 multiply-stores per row / column; all four
// targets of a half-step accumulate in one pass (each scale loaded once) and their reciprocals run
// as two interleaved Newton pairs. Same math (exact reciprocal: arecip seed + 2 Newton steps),
// different rounding sequence: hc comb rel-L2 vs fp64 7.9e-8 (was 9.3e-8), pre / post unchanged;
// 5.48 -> 3.49 us per hc row (perf_experiments/hc_sfpu, V11). Slots: u_j / v_k at the scratch
// quantities 24..31 (ss is dead after the logits; 25..31 are unused lanes), P / Q in the 8 vectors
// after the lane tile (dst_reg[32..39], DEST tile D1 — free in the NATURAL hc_row).
constexpr uint32_t u_q(uint32_t j) { return q(mhc_post_hc::SS_Q + j); }
constexpr uint32_t v_q(uint32_t k) { return q(mhc_post_hc::SS_Q + mhc_post_hc::N + k); }
constexpr uint32_t P_SLOT = 32;  // P_j
constexpr uint32_t Q_SLOT = 36;  // Q_k
#define MHC_HC_EPS sfpi::vConstFloatPrgm1

// 1 / x for finite x > 0 (sums >= eps): arecip seed + 2 Newton steps in residual form (no NaN guard,
// no -0.0f addends; same seed / steps as sfpu_reciprocal_iter<2>).
sfpi_inline sfpi::vFloat recip_n(sfpi::vFloat x) {
    sfpi::vFloat y = sfpi::approx_recip(x);
    sfpi::vFloat e = 1.0f - x * y;
    y = y * e + y;
    e = 1.0f - x * y;
    y = y * e + y;
    return y;
}

sfpi_inline void recip_pair(sfpi::vFloat d0, sfpi::vFloat d1, uint32_t o0, uint32_t o1) {
    sfpi::vFloat y0 = sfpi::approx_recip(d0);
    sfpi::vFloat y1 = sfpi::approx_recip(d1);
    sfpi::vFloat e0 = 1.0f - d0 * y0;
    sfpi::vFloat e1 = 1.0f - d1 * y1;
    y0 = y0 * e0 + y0;
    y1 = y1 * e1 + y1;
    e0 = 1.0f - d0 * y0;
    e1 = 1.0f - d1 * y1;
    y0 = y0 * e0 + y0;
    y1 = y1 * e1 + y1;
    sfpi::dst_reg[o0] = y0;
    sfpi::dst_reg[o1] = y1;
}

template <bool COL>
constexpr uint32_t kidx(uint32_t t, uint32_t k) {
    return COL ? m(k, t) : m(t, k);
}
template <bool COL>
constexpr uint32_t src_q(uint32_t k) {
    return COL ? u_q(k) : v_q(k);
}
template <bool COL>
constexpr uint32_t tgt_q(uint32_t t) {
    return COL ? v_q(t) : u_q(t);
}

// One half-step: COL = column step (targets v from u), else row step (targets u from v).
template <bool COL>
sfpi_inline void scale_quad() {
    constexpr uint32_t n = mhc_post_hc::N;
    constexpr uint32_t REC = COL ? Q_SLOT : P_SLOT;
    sfpi::vFloat w = ld(src_q<COL>(0));
    sfpi::vFloat a0 = ld(kidx<COL>(0, 0)) * w;
    sfpi::vFloat a1 = ld(kidx<COL>(1, 0)) * w;
    sfpi::vFloat a2 = ld(kidx<COL>(2, 0)) * w;
    sfpi::vFloat a3 = ld(kidx<COL>(3, 0)) * w;
#pragma GCC unroll 3
    for (uint32_t k = 1; k < n; ++k) {
        w = ld(src_q<COL>(k));
        a0 = ld(kidx<COL>(0, k)) * w + a0;
        a1 = ld(kidx<COL>(1, k)) * w + a1;
        a2 = ld(kidx<COL>(2, k)) * w + a2;
        a3 = ld(kidx<COL>(3, k)) * w + a3;
    }
    a0 = ld(REC + 0) * MHC_HC_EPS + a0;
    a1 = ld(REC + 1) * MHC_HC_EPS + a1;
    a2 = ld(REC + 2) * MHC_HC_EPS + a2;
    a3 = ld(REC + 3) * MHC_HC_EPS + a3;
    sfpi::dst_reg[REC + 0] = a0;
    sfpi::dst_reg[REC + 1] = a1;
    sfpi::dst_reg[REC + 2] = a2;
    sfpi::dst_reg[REC + 3] = a3;
    recip_pair(a0, a1, tgt_q<COL>(0), tgt_q<COL>(1));
    recip_pair(a2, a3, tgt_q<COL>(2), tgt_q<COL>(3));
}

// The whole Sinkhorn (column step, then iters - 1 row / column pairs) on K in the M slots.
sfpi_inline void sinkhorn_scaled(uint32_t iters, uint32_t hc_eps_bits) {
    constexpr uint32_t n = mhc_post_hc::N;
    MHC_HC_EPS = Converter::as_float(hc_eps_bits);
#pragma GCC unroll 4
    for (uint32_t k = 0; k < n; ++k) {
        sfpi::vFloat s0 = ld(m(0, k)) + ld(m(1, k));
        sfpi::vFloat s1 = ld(m(2, k)) + ld(m(3, k));
        sfpi::vFloat s = (s0 + s1) + MHC_HC_EPS;
        sfpi::dst_reg[Q_SLOT + k] = s;
        sfpi::dst_reg[v_q(k)] = recip_n(s);
    }
    if (iters > 1) {
#pragma GCC unroll 4
        for (uint32_t j = 0; j < n; ++j) {
            sfpi::dst_reg[P_SLOT + j] = 1.0f;
        }
#pragma GCC unroll 0
        for (uint32_t it = 1; it < iters; ++it) {
            scale_quad<false>();
            scale_quad<true>();
        }
#pragma GCC unroll 4
        for (uint32_t j = 0; j < n; ++j) {
            sfpi::vFloat u = ld(u_q(j));
#pragma GCC unroll 4
            for (uint32_t k = 0; k < n; ++k) {
                sfpi::dst_reg[m(j, k)] = (sfpi::dst_reg[m(j, k)] * u) * sfpi::dst_reg[v_q(k)];
            }
        }
    } else {
#pragma GCC unroll 4
        for (uint32_t j = 0; j < n; ++j) {
#pragma GCC unroll 4
            for (uint32_t k = 0; k < n; ++k) {
                sfpi::dst_reg[m(j, k)] = sfpi::dst_reg[m(j, k)] * sfpi::dst_reg[v_q(k)];
            }
        }
    }
}
#undef MHC_HC_EPS

}  // namespace mhc_hc_detail

// The whole hc program over the lane tiles starting at the current dest base (RC_custom: called
// once). `p` = the mhc_post_hc parameter block.
inline void calculate_mhc_hc(const uint32_t* p) {
    using namespace mhc_hc_detail;
    constexpr uint32_t n = mhc_post_hc::N;
    sfpi::vConstFloatPrgm0 = 2.0f;  // Newton constant of sfpu_reciprocal_iter
    const uint32_t iters = p[mhc_post_hc::P_ITERS];
    const uint32_t hc_eps = p[mhc_post_hc::P_HC_EPS];

#pragma GCC unroll 0
    for (uint32_t half = 0; half < mhc_post_hc::HALVES; ++half) {
        // r = rsqrt(ss / width + norm_eps), kept in the ss slot
        {
            sfpi::vFloat ss = sfpi::dst_reg[q(mhc_post_hc::SS_Q)];
            sfpi::vFloat x = ss * bits(p[mhc_post_hc::P_INV_WIDTH]) + bits(p[mhc_post_hc::P_NORM_EPS]);
            sfpi::dst_reg[q(mhc_post_hc::SS_Q)] = rsqrt(x);
        }
        // pre / post
#pragma GCC unroll 4
        for (uint32_t k = 0; k < n; ++k) {
            sfpi::vFloat z = ld(q(k)) * ld(q(mhc_post_hc::SS_Q));
            sfpi::vFloat t = z * bits(p[mhc_post_hc::P_A_PRE]) + bits(p[mhc_post_hc::P_BASE + k]);
            sfpi::dst_reg[q(k)] = sigmoid(t) + bits(hc_eps);
        }
#pragma GCC unroll 4
        for (uint32_t k = n; k < 2 * n; ++k) {
            sfpi::vFloat z = ld(q(k)) * ld(q(mhc_post_hc::SS_Q));
            sfpi::vFloat t = z * bits(p[mhc_post_hc::P_A_POST]) + bits(p[mhc_post_hc::P_BASE + k]);
            sfpi::vFloat s = sigmoid(t);
            sfpi::dst_reg[q(k)] = s + s;
        }
        // logits + row softmax (row max subtracted) + hc_eps (unrolled: dst_reg indices stay immediates)
#pragma GCC unroll 4
        for (uint32_t j = 0; j < n; ++j) {
            sfpi::vFloat mx = 0.0f;
#pragma GCC unroll 4
            for (uint32_t k = 0; k < n; ++k) {
                const uint32_t qq = 2 * n + j * n + k;
                sfpi::vFloat z = ld(m(j, k)) * ld(q(mhc_post_hc::SS_Q));
                sfpi::vFloat l = z * bits(p[mhc_post_hc::P_A_RES]) + bits(p[mhc_post_hc::P_BASE + qq]);
                sfpi::dst_reg[m(j, k)] = l;
                if (k == 0) {
                    mx = l;
                } else {
                    v_if(l > mx) { mx = l; }
                    v_endif;
                }
            }
#pragma GCC unroll 4
            for (uint32_t k = 0; k < n; ++k) {
                sfpi::dst_reg[m(j, k)] = sfpi::dst_reg[m(j, k)] - mx;
            }
#pragma GCC unroll 4
            for (uint32_t k = 0; k < n; ++k) {
                sfpi::dst_reg[m(j, k)] = _sfpu_exp_fp32_accurate_<false>(sfpi::dst_reg[m(j, k)]);
            }
            sfpi::vFloat s = ld(m(j, 0)) + ld(m(j, 1));
            s = s + sfpi::dst_reg[m(j, 2)];
            s = s + sfpi::dst_reg[m(j, 3)];
            sfpi::vFloat inv = recip(s);
#pragma GCC unroll 4
            for (uint32_t k = 0; k < n; ++k) {
                sfpi::dst_reg[m(j, k)] = sfpi::dst_reg[m(j, k)] * inv + bits(hc_eps);
            }
        }
        // Sinkhorn: column step, then iters - 1 (row, column) pairs
        if constexpr (mhc_post_hc::NATURAL) {
            sinkhorn_scaled(iters, hc_eps);
        } else {
            // ROBUST (non-Blackhole) carve-out: its two lane tiles occupy D0 and D1, so the scaled
            // form's P / Q scratch (dst_reg[32..39]) is not free there.
            col_normalize(hc_eps);
#pragma GCC unroll 0
            for (uint32_t it = 1; it < iters; ++it) {
                row_normalize(hc_eps);
                col_normalize(hc_eps);
            }
        }
        if constexpr (!mhc_post_hc::NATURAL) {
            sfpi::dst_reg++;  // ROBUST: the second half (vectors 2k + 1)
        }
    }
}

// NATURAL: row group m (tile rows 4m..4m+3 of the transposed stats tile) <-> one quantity per
// vector. SFPTRANSP is an involution, so the same sequence regroups and restores. One out-of-line
// loop (runtime-encoded TT_ instructions) shared by gather and scatter: the profiler build's
// kernel-text budget has no room for 13 unrolled copies.
__attribute__((noinline)) void mhc_hc_regroup(uint32_t groups) {
#pragma GCC unroll 0
    for (uint32_t m = 0; m < groups; ++m) {
        const uint32_t base = mhc_post_hc::nat_base(m);
        TT_SFPLOAD(p_sfpu::LREG0, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_7, base + 0);
        TT_SFPLOAD(p_sfpu::LREG1, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_7, base + 2);
        TT_SFPLOAD(p_sfpu::LREG2, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_7, base + 32);
        TT_SFPLOAD(p_sfpu::LREG3, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_7, base + 34);
        TTI_SFPTRANSP(0, 0, 0, 0);
        TT_SFPSTORE(p_sfpu::LREG0, sfpi::SFPSTORE_MOD0_FMT_FP32, ADDR_MOD_7, base + 0);
        TT_SFPSTORE(p_sfpu::LREG1, sfpi::SFPSTORE_MOD0_FMT_FP32, ADDR_MOD_7, base + 2);
        TT_SFPSTORE(p_sfpu::LREG2, sfpi::SFPSTORE_MOD0_FMT_FP32, ADDR_MOD_7, base + 32);
        TT_SFPSTORE(p_sfpu::LREG3, sfpi::SFPSTORE_MOD0_FMT_FP32, ADDR_MOD_7, base + 34);
    }
}

inline void mhc_hc_gather() { mhc_hc_regroup(7); }  // quantities 0 .. 27 (24 = ss)

inline void mhc_hc_scatter() {
    using namespace mhc_post_hc;
    mhc_hc_regroup(6);  // quantities 0 .. 23
    // hc page columns 24..31 are zero: quantity rows 24..31 (groups 6, 7)
    sfpi::vFloat z = 0.0f;
#pragma GCC unroll 8
    for (uint32_t m = 6; m < 8; ++m) {
#pragma GCC unroll 4
        for (uint32_t i = 0; i < 4; ++i) {
            sfpi::dst_reg[(nat_base(m) + nat_off(i)) / 2] = z;
        }
    }
}

inline void calculate_mhc_hc_natural(const uint32_t* p) {
    mhc_hc_gather();
    calculate_mhc_hc(p);
    mhc_hc_scatter();
}

}  // namespace ckernel::sfpu
#endif

#ifdef MHC_POST_HC_COMPUTE
namespace mhc_post_hc {

// eltwise_chain element: runs the hc program in place over lane tiles Slot, Slot + 1.
template <compute_kernel_lib::Dst Slot>
struct HcProgram : compute_kernel_lib::UnaryOp<HcProgram<Slot>, Slot> {
    static constexpr uint32_t lane_width = compute_kernel_lib::to_u32(Slot) + LANE_TILES;
    const uint32_t* params;

    ALWI explicit HcProgram(const uint32_t* p) : params(p) {}

    static ALWI void init() {
        MATH((::ckernel::llk_math_eltwise_unary_sfpu_init<::SfpuType::unused, DST_ACCUM_MODE>()));
    }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            ::ckernel::sfpu::calculate_mhc_hc,
            compute_kernel_lib::to_u32(Slot) + slot_offset,
            VectorMode::RC_custom,
            params)));
    }
};

}  // namespace mhc_post_hc
#endif
