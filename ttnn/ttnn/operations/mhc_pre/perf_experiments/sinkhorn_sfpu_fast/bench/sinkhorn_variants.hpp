// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Sinkhorn variants on one coefficient-major fp32 DEST tile (slot k = dst_reg[k], lane = token; logits /
// comb in slots [LOGIT0, LOGIT0 + N*N), m[i][j] = slot LOGIT0 + i*N + j). Included on TRISC_MATH only.
#pragma once

namespace skb {

using namespace sfpi;
using ckernel::sfpu::Converter;

constexpr int N = static_cast<int>(n_streams);
constexpr int LOGIT0 = 2 * N;
constexpr int SCRATCH0 = 31;
constexpr int SCRATCH1 = 30;

// ============================ v0: the op's current code, verbatim ============================
namespace v0 {
sfpi_inline vFloat recip_pos(vFloat x) {
    vFloat y = approx_recip(x);
    vFloat t = 2.0f - x * y;
    y = y * t;
    t = 2.0f - x * y;
    y = y * t;
    return y;
}

sfpi_inline void col_norm(uint32_t eps_bits) {
#pragma GCC unroll 8
    for (int j = 0; j < N; ++j) {
        vFloat s = dst_reg[LOGIT0 + j];
        for (int i = 1; i < N; ++i) {
            s = s + dst_reg[LOGIT0 + i * N + j];
        }
        vFloat e = Converter::as_float(eps_bits);
        vFloat rc = recip_pos(s + e);
        for (int i = 0; i < N; ++i) {
            dst_reg[LOGIT0 + i * N + j] = dst_reg[LOGIT0 + i * N + j] * rc;
        }
    }
}

sfpi_inline void row_norm(uint32_t eps_bits) {
#pragma GCC unroll 8
    for (int i = 0; i < N; ++i) {
        vFloat s = dst_reg[LOGIT0 + i * N];
        for (int j = 1; j < N; ++j) {
            s = s + dst_reg[LOGIT0 + i * N + j];
        }
        vFloat e = Converter::as_float(eps_bits);
        vFloat rr = recip_pos(s + e);
        for (int j = 0; j < N; ++j) {
            dst_reg[LOGIT0 + i * N + j] = dst_reg[LOGIT0 + i * N + j] * rr;
        }
    }
}

void sinkhorn(uint32_t eps_bits, uint32_t iters) {
#pragma GCC unroll 8
    for (int i = 0; i < N; ++i) {
        const int row = LOGIT0 + i * N;
        {
            vFloat mx = dst_reg[row];
            for (int j = 1; j < N; ++j) {
                vFloat t = dst_reg[row + j];
                v_if(t > mx) { mx = t; }
                v_endif;
            }
            dst_reg[SCRATCH0] = mx;
            dst_reg[SCRATCH1] = 0.0f;
        }
        for (int j = 0; j < N; ++j) {
            vFloat lv = dst_reg[row + j];
            vFloat mxv = dst_reg[SCRATCH0];
            vFloat d = lv - mxv;
            vFloat ex = ckernel::sfpu::_sfpu_exp_fp32_accurate_<false>(d);
            dst_reg[row + j] = ex;
            dst_reg[SCRATCH1] = dst_reg[SCRATCH1] + ex;
        }
        {
            vFloat rs = recip_pos(dst_reg[SCRATCH1]);
            vFloat e = Converter::as_float(eps_bits);
            for (int j = 0; j < N; ++j) {
                dst_reg[row + j] = dst_reg[row + j] * rs + e;
            }
        }
    }
    col_norm(eps_bits);
#pragma GCC unroll 0
    for (uint32_t it = 1; it < iters; ++it) {
        row_norm(eps_bits);
        col_norm(eps_bits);
    }
}
}  // namespace v0

// Shared by v1+: 2.0 (Newton) and eps live in the programmable constant registers (set once per call),
// so no SFPLOADI per use. Same instructions otherwise -> bitwise identical.
constexpr int MIX = N * (N + 2);
constexpr int SCR = MIX + 1;  // free slots MIX+1 .. 31 (slot MIX = r, dead after the coefficients)
constexpr int M(int i, int j) { return LOGIT0 + i * N + j; }

sfpi_inline void set_consts(uint32_t eps_bits) {
    vConstFloatPrgm0 = 2.0f;
    vConstFloatPrgm1 = Converter::as_float(eps_bits);
}
#define SKB_TWO vConstFloatPrgm0
#define SKB_EPS vConstFloatPrgm1

sfpi_inline vFloat recip_c(vFloat x) {
    vFloat y = approx_recip(x);
    vFloat t = SKB_TWO - x * y;
    y = y * t;
    t = SKB_TWO - x * y;
    y = y * t;
    return y;
}

// two independent reciprocals, interleaved (same per-chain op sequence as recip_c)
sfpi_inline void recip2(vFloat& a, vFloat& b) {
    vFloat ya = approx_recip(a);
    vFloat yb = approx_recip(b);
    vFloat ta = SKB_TWO - a * ya;
    vFloat tb = SKB_TWO - b * yb;
    ya = ya * ta;
    yb = yb * tb;
    ta = SKB_TWO - a * ya;
    tb = SKB_TWO - b * yb;
    a = ya * ta;
    b = yb * tb;
}

// softmax (row max subtracted) + eps, as v0 (same op order), max / sum kept in LREGs
sfpi_inline void softmax_rows() {
#pragma GCC unroll 8
    for (int i = 0; i < N; ++i) {
        vFloat mx = dst_reg[M(i, 0)];
        for (int j = 1; j < N; ++j) {
            vFloat t = dst_reg[M(i, j)];
            v_if(t > mx) { mx = t; }
            v_endif;
        }
        vFloat sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            vFloat d = dst_reg[M(i, j)] - mx;
            vFloat ex = ckernel::sfpu::_sfpu_exp_fp32_accurate_<false>(d);
            dst_reg[M(i, j)] = ex;
            sum = sum + ex;
        }
        vFloat rs = recip_c(sum);
        for (int j = 0; j < N; ++j) {
            dst_reg[M(i, j)] = dst_reg[M(i, j)] * rs + SKB_EPS;
        }
    }
}

// ============================ v1: v0 + constant registers ============================
namespace v1 {
sfpi_inline void col_norm() {
#pragma GCC unroll 8
    for (int j = 0; j < N; ++j) {
        vFloat s = dst_reg[M(0, j)];
        for (int i = 1; i < N; ++i) {
            s = s + dst_reg[M(i, j)];
        }
        vFloat rc = recip_c(s + SKB_EPS);
        for (int i = 0; i < N; ++i) {
            dst_reg[M(i, j)] = dst_reg[M(i, j)] * rc;
        }
    }
}
sfpi_inline void row_norm() {
#pragma GCC unroll 8
    for (int i = 0; i < N; ++i) {
        vFloat s = dst_reg[M(i, 0)];
        for (int j = 1; j < N; ++j) {
            s = s + dst_reg[M(i, j)];
        }
        vFloat rr = recip_c(s + SKB_EPS);
        for (int j = 0; j < N; ++j) {
            dst_reg[M(i, j)] = dst_reg[M(i, j)] * rr;
        }
    }
}
void sinkhorn(uint32_t eps_bits, uint32_t iters) {
    set_consts(eps_bits);
    softmax_rows();
    col_norm();
#pragma GCC unroll 0
    for (uint32_t it = 1; it < iters; ++it) {
        row_norm();
        col_norm();
    }
}
}  // namespace v1

// ============================ v2: v1 + interleaved chains (no pass fusion) ============================
namespace v2 {
// column (COL) or row sums of 4 lines, 4 accumulators interleaved; same per-line add order as v0
template <bool COL>
sfpi_inline void sums4(vFloat& s0, vFloat& s1, vFloat& s2, vFloat& s3) {
    auto at = [](int line, int k) { return COL ? M(k, line) : M(line, k); };
    s0 = dst_reg[at(0, 0)];
    s1 = dst_reg[at(1, 0)];
    s2 = dst_reg[at(2, 0)];
    s3 = dst_reg[at(3, 0)];
#pragma GCC unroll 4
    for (int k = 1; k < N; ++k) {
        s0 = s0 + dst_reg[at(0, k)];
        s1 = s1 + dst_reg[at(1, k)];
        s2 = s2 + dst_reg[at(2, k)];
        s3 = s3 + dst_reg[at(3, k)];
    }
}
template <bool COL>
sfpi_inline void norm() {
    vFloat s0, s1, s2, s3;
    sums4<COL>(s0, s1, s2, s3);
    s0 = s0 + SKB_EPS;
    s1 = s1 + SKB_EPS;
    recip2(s0, s1);
    s2 = s2 + SKB_EPS;
    s3 = s3 + SKB_EPS;
    recip2(s2, s3);
    // scale: 4 independent load-mul-store per step
#pragma GCC unroll 4
    for (int k = 0; k < N; ++k) {
        if constexpr (COL) {
            vFloat a = dst_reg[M(k, 0)] * s0;
            vFloat b = dst_reg[M(k, 1)] * s1;
            vFloat c = dst_reg[M(k, 2)] * s2;
            vFloat d = dst_reg[M(k, 3)] * s3;
            dst_reg[M(k, 0)] = a;
            dst_reg[M(k, 1)] = b;
            dst_reg[M(k, 2)] = c;
            dst_reg[M(k, 3)] = d;
        } else {
            vFloat a = dst_reg[M(0, k)] * s0;
            vFloat b = dst_reg[M(1, k)] * s1;
            vFloat c = dst_reg[M(2, k)] * s2;
            vFloat d = dst_reg[M(3, k)] * s3;
            dst_reg[M(0, k)] = a;
            dst_reg[M(1, k)] = b;
            dst_reg[M(2, k)] = c;
            dst_reg[M(3, k)] = d;
        }
    }
}
void sinkhorn(uint32_t eps_bits, uint32_t iters) {
    set_consts(eps_bits);
    softmax_rows();
    norm<true>();
#pragma GCC unroll 0
    for (uint32_t it = 1; it < iters; ++it) {
        norm<false>();
        norm<true>();
    }
}
}  // namespace v2

// ============================ v3: fused passes (scale + the other direction's sums) ============================
// State between passes: the column sums c_j (after a row scaling) or the row reciprocals rr_i (DEST scratch).
//   pass C: m_ij *= rc_j (store), row sums s_i = ((v_i0 + v_i1) + v_i2) + v_i3 -> scratch
//   pass R: m_ij *= rr_i (store), column sums c_j accumulated in row order (= v0's i order)
// Every element sees exactly v0's multiplications and every sum v0's operand order -> bitwise identical,
// minus v0's separate 16-load sum pass per normalisation.
namespace v3 {
sfpi_inline void recip4_from(vFloat& c0, vFloat& c1, vFloat& c2, vFloat& c3) {
    c0 = c0 + SKB_EPS;
    c1 = c1 + SKB_EPS;
    recip2(c0, c1);
    c2 = c2 + SKB_EPS;
    c3 = c3 + SKB_EPS;
    recip2(c2, c3);
}
// col scale with rc_j; if ROWSUMS, row sums -> scratch SCR + i
template <bool ROWSUMS>
sfpi_inline void pass_c(vFloat rc0, vFloat rc1, vFloat rc2, vFloat rc3) {
#pragma GCC unroll 4
    for (int i = 0; i < N; ++i) {
        vFloat a = dst_reg[M(i, 0)] * rc0;
        vFloat b = dst_reg[M(i, 1)] * rc1;
        dst_reg[M(i, 0)] = a;
        dst_reg[M(i, 1)] = b;
        vFloat s = a + b;
        a = dst_reg[M(i, 2)] * rc2;
        b = dst_reg[M(i, 3)] * rc3;
        dst_reg[M(i, 2)] = a;
        dst_reg[M(i, 3)] = b;
        if constexpr (ROWSUMS) {
            s = s + a;
            s = s + b;
            dst_reg[SCR + i] = s;
        }
    }
}
// row reciprocals from the scratch row sums, in place
sfpi_inline void row_recips() {
    vFloat s0 = dst_reg[SCR + 0];
    vFloat s1 = dst_reg[SCR + 1];
    vFloat s2 = dst_reg[SCR + 2];
    vFloat s3 = dst_reg[SCR + 3];
    recip4_from(s0, s1, s2, s3);
    dst_reg[SCR + 0] = s0;
    dst_reg[SCR + 1] = s1;
    dst_reg[SCR + 2] = s2;
    dst_reg[SCR + 3] = s3;
}
// row scale with rr_i (scratch), column sums -> c_j
sfpi_inline void pass_r(vFloat& c0, vFloat& c1, vFloat& c2, vFloat& c3) {
    {
        vFloat rr = dst_reg[SCR + 0];
        c0 = dst_reg[M(0, 0)] * rr;
        c1 = dst_reg[M(0, 1)] * rr;
        c2 = dst_reg[M(0, 2)] * rr;
        c3 = dst_reg[M(0, 3)] * rr;
        dst_reg[M(0, 0)] = c0;
        dst_reg[M(0, 1)] = c1;
        dst_reg[M(0, 2)] = c2;
        dst_reg[M(0, 3)] = c3;
    }
#pragma GCC unroll 4
    for (int i = 1; i < N; ++i) {
        vFloat rr = dst_reg[SCR + i];
        vFloat a = dst_reg[M(i, 0)] * rr;
        vFloat b = dst_reg[M(i, 1)] * rr;
        dst_reg[M(i, 0)] = a;
        dst_reg[M(i, 1)] = b;
        c0 = c0 + a;
        c1 = c1 + b;
        a = dst_reg[M(i, 2)] * rr;
        b = dst_reg[M(i, 3)] * rr;
        dst_reg[M(i, 2)] = a;
        dst_reg[M(i, 3)] = b;
        c2 = c2 + a;
        c3 = c3 + b;
    }
}
sfpi_inline void col_sums(vFloat& c0, vFloat& c1, vFloat& c2, vFloat& c3) { v2::sums4<true>(c0, c1, c2, c3); }

void sinkhorn(uint32_t eps_bits, uint32_t iters) {
    set_consts(eps_bits);
    softmax_rows();
    vFloat c0, c1, c2, c3;
    col_sums(c0, c1, c2, c3);
#pragma GCC unroll 0
    for (uint32_t it = 1; it < iters; ++it) {
        recip4_from(c0, c1, c2, c3);
        pass_c<true>(c0, c1, c2, c3);
        row_recips();
        pass_r(c0, c1, c2, c3);
    }
    recip4_from(c0, c1, c2, c3);
    pass_c<false>(c0, c1, c2, c3);
}
}  // namespace v3

// ============================ v4 / v5: v3 + a cheaper softmax ============================
// v4: softmax j loops fully unrolled (v3's exp loop was a runtime loop: its dst addresses were injected
//     instruction words); v5: + the row max via SFPSWAP min/max (sfpi::max) instead of v_if compare chains.
template <bool SWAPMAX>
sfpi_inline void softmax_rows_unrolled() {
#pragma GCC unroll 4
    for (int i = 0; i < N; ++i) {
        vFloat mx = dst_reg[M(i, 0)];
#pragma GCC unroll 4
        for (int j = 1; j < N; ++j) {
            vFloat t = dst_reg[M(i, j)];
            if constexpr (SWAPMAX) {
                mx = sfpi::max(mx, t);
            } else {
                v_if(t > mx) { mx = t; }
                v_endif;
            }
        }
        vFloat sum = 0.0f;
#pragma GCC unroll 4
        for (int j = 0; j < N; ++j) {
            vFloat d = dst_reg[M(i, j)] - mx;
            vFloat ex = ckernel::sfpu::_sfpu_exp_fp32_accurate_<false>(d);
            dst_reg[M(i, j)] = ex;
            sum = sum + ex;
        }
        vFloat rs = recip_c(sum);
#pragma GCC unroll 4
        for (int j = 0; j < N; ++j) {
            dst_reg[M(i, j)] = dst_reg[M(i, j)] * rs + SKB_EPS;
        }
    }
}
template <bool SWAPMAX>
void sinkhorn_v45(uint32_t eps_bits, uint32_t iters) {
    set_consts(eps_bits);
    softmax_rows_unrolled<SWAPMAX>();
    vFloat c0, c1, c2, c3;
    v3::col_sums(c0, c1, c2, c3);
#pragma GCC unroll 0
    for (uint32_t it = 1; it < iters; ++it) {
        v3::recip4_from(c0, c1, c2, c3);
        v3::pass_c<true>(c0, c1, c2, c3);
        v3::row_recips();
        v3::pass_r(c0, c1, c2, c3);
    }
    v3::recip4_from(c0, c1, c2, c3);
    v3::pass_c<false>(c0, c1, c2, c3);
}

// ============================ v6: v5 + hand-scheduled SFPLOADMACRO passes (raw TTI) ============================
// Raw-LLK: the per-iteration loop is hand-written TTI (no sfpi) because the macro passes pin registers
// (L0..L2 macro temps, L3 accumulator, L4..L7 multipliers) that the sfpi register allocator does not know about.
// Same math, same operand order, same instruction semantics as v0 (SFPMUL / SFPADD = MAD with L9 = 0 / L10 = 1).
}  // namespace skb
#include "sinkhorn_lm_gen.hpp"
namespace skb {
namespace v6 {
constexpr uint32_t SCRA = 2 * SCR;  // DEST address of scratch slot SCR
// SFPLOADMACRO config: macro q (q = 0..3): MAD sub-unit = template q = SFPMUL(VA = L(4+q), VB <- the loaded VD,
// VC = L9 = 0), delay 0; store sub-unit = SFPSTORE of VD to the loaded address, 2 issued instructions later.
sfpi_inline void skm_config() {
    TTI_SFPMUL(4, 0, 9, 12, 0);  // VD = 12 + q: backdoor write of InstructionTemplate[q]
    TTI_SFPMUL(5, 0, 9, 13, 0);
    TTI_SFPMUL(6, 0, 9, 14, 0);
    TTI_SFPMUL(7, 0, 9, 15, 0);
    constexpr uint32_t store_bits = (2 << 3) | 3;
#define SKM_SEQ(q)                                                                        \
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, ((0x80 | (0 << 3) | (4 + (q))) << 8) | 0); \
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | 0);                    \
    TTI_SFPCONFIG(0, 4 + (q), 0);
    SKM_SEQ(0)
    SKM_SEQ(1)
    SKM_SEQ(2)
    SKM_SEQ(3)
#undef SKM_SEQ
    // Misc: UsesLoadMod0ForStore = 1 for macros 0..3, UnitDelayKind = WaitForElapsedInstructions for MAD + store
    TTI_SFPCONFIG(0xAF0, 8, 1);
}
// L4..L7 <- 1 / (x_k + eps), the recip_c Newton sequence, two chains interleaved (temps L0..L3).
// FROM_ACC: x_3 is the last group sum still in L3 (the pass does not store it); x_0..x_2 (x_0..x_3 otherwise) come
// from scratch, >= 8 issued instructions after their stores (Dst write -> SFPLOAD needs > 4 cycles, not interlocked).
#define SKM_RECIP2(a, b, EPS_B)      \
    TTI_SFPADD(10, a, 13, a, 0);     \
    if (EPS_B) {                     \
        TTI_SFPADD(10, b, 13, b, 0); \
    }                                \
    TTI_SFPARECIP(0, a, 0, 0);       \
    TTI_SFPARECIP(0, b, 1, 0);       \
    TTI_SFPMAD(a, 0, 12, 2, 1);      \
    TTI_SFPMAD(b, 1, 12, 3, 1);      \
    TTI_SFPMUL(0, 2, 9, 0, 0);       \
    TTI_SFPMUL(1, 3, 9, 1, 0);       \
    TTI_SFPMAD(a, 0, 12, 2, 1);      \
    TTI_SFPMAD(b, 1, 12, 3, 1);      \
    TTI_SFPMUL(0, 2, 9, a, 0);       \
    TTI_SFPMUL(1, 3, 9, b, 0);
template <bool FROM_ACC>
sfpi_inline void skm_recips() {
    if constexpr (FROM_ACC) {
        TTI_SFPADD(10, 3, 13, 7, 0);  // L7 = acc + eps, before L3 is reused as a temp
    }
    TTI_SFPLOAD(4, 0, 7, SCRA + 0);
    TTI_SFPLOAD(5, 0, 7, SCRA + 2);
    TTI_SFPLOAD(6, 0, 7, SCRA + 4);
    if constexpr (!FROM_ACC) {
        TTI_SFPLOAD(7, 0, 7, SCRA + 6);
    }
    SKM_RECIP2(4, 5, true)
    SKM_RECIP2(6, 7, !FROM_ACC)
}
#undef SKM_RECIP2
void sinkhorn(uint32_t eps_bits, uint32_t iters) {
    set_consts(eps_bits);
    softmax_rows_unrolled<true>();
    {
        vFloat c0, c1, c2, c3;
        v3::col_sums(c0, c1, c2, c3);
        dst_reg[SCR + 0] = c0;
        dst_reg[SCR + 1] = c1;
        dst_reg[SCR + 2] = c2;
        dst_reg[SCR + 3] = c3;
    }
    skm_config();
    skm_recips<false>();  // rc_j of the first col_norm
#pragma GCC unroll 0
    for (uint32_t it = 1; it < iters; ++it) {
        skm_pass_c_sums();
        skm_recips<true>();  // rr_i
        skm_pass_r_sums();
        skm_recips<true>();  // rc_j
    }
    skm_pass_c_final();
    TTI_SFPNOP;  // the last macro stores land > 4 cycles before any following Dst read
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}
}  // namespace v6

// ============================ v7: scaling-vector form (NOT bitwise; algebraically identical)
// ============================ m = diag(r) K diag(c), K = softmax + eps fixed in DEST; row_norm: r_i <- r_i / (r_i (K
// c)_i + eps); col_norm: c_j <- c_j / (c_j (r^T K)_j + eps); (K c)_i as a fused MAD chain in j order. c in LREGs during
// the row update, r in scratch slots SCR..SCR+3; the column update keeps r in LREGs and c in scratch SCR+4..SCR+7 (<=
// slot 31 + ...).
namespace v7 {
constexpr int RS = SCR;      // r_i scratch
constexpr int CS = SCR + 4;  // c_j scratch  (SCR + 7 = 32 > 31 for N = 4: use slots 24.. instead)
}  // namespace v7
namespace v7b {
constexpr int RS = MIX;      // slots 24..27 (slot MIX = r of the coefficients: dead after them)
constexpr int CS = MIX + 4;  // slots 28..31
sfpi_inline vFloat rowdot(int i, vFloat c0, vFloat c1, vFloat c2, vFloat c3) {
    vFloat t = dst_reg[M(i, 0)] * c0;
    t = dst_reg[M(i, 1)] * c1 + t;
    t = dst_reg[M(i, 2)] * c2 + t;
    t = dst_reg[M(i, 3)] * c3 + t;
    return t;
}
sfpi_inline vFloat coldot(int j, vFloat r0, vFloat r1, vFloat r2, vFloat r3) {
    vFloat t = dst_reg[M(0, j)] * r0;
    t = dst_reg[M(1, j)] * r1 + t;
    t = dst_reg[M(2, j)] * r2 + t;
    t = dst_reg[M(3, j)] * r3 + t;
    return t;
}
// v <- v / (v * t + eps)
sfpi_inline vFloat upd(vFloat v, vFloat t) { return v * recip_c(v * t + SKB_EPS); }
void sinkhorn(uint32_t eps_bits, uint32_t iters) {
    set_consts(eps_bits);
    softmax_rows_unrolled<true>();
    vFloat c0, c1, c2, c3;
    v3::col_sums(c0, c1, c2, c3);
    v3::recip4_from(c0, c1, c2, c3);  // c = 1 / (colsum(K) + eps), r = 1
#pragma GCC unroll 4
    for (int i = 0; i < N; ++i) {
        dst_reg[RS + i] = 1.0f;
    }
#pragma GCC unroll 0
    for (uint32_t it = 1; it < iters; ++it) {
        // row update (c in LREGs)
#pragma GCC unroll 4
        for (int i = 0; i < N; ++i) {
            vFloat t = rowdot(i, c0, c1, c2, c3);
            dst_reg[RS + i] = upd(dst_reg[RS + i], t);
        }
        dst_reg[CS + 0] = c0;
        dst_reg[CS + 1] = c1;
        dst_reg[CS + 2] = c2;
        dst_reg[CS + 3] = c3;
        // column update (r in LREGs)
        vFloat r0 = dst_reg[RS + 0], r1 = dst_reg[RS + 1], r2 = dst_reg[RS + 2], r3 = dst_reg[RS + 3];
#pragma GCC unroll 4
        for (int j = 0; j < N; ++j) {
            vFloat t = coldot(j, r0, r1, r2, r3);
            dst_reg[CS + j] = upd(dst_reg[CS + j], t);
        }
        c0 = dst_reg[CS + 0];
        c1 = dst_reg[CS + 1];
        c2 = dst_reg[CS + 2];
        c3 = dst_reg[CS + 3];
    }
    // m_ij = (r_i K_ij) c_j
#pragma GCC unroll 4
    for (int i = 0; i < N; ++i) {
        vFloat r = dst_reg[RS + i];
        dst_reg[M(i, 0)] = dst_reg[M(i, 0)] * r * c0;
        dst_reg[M(i, 1)] = dst_reg[M(i, 1)] * r * c1;
        dst_reg[M(i, 2)] = dst_reg[M(i, 2)] * r * c2;
        dst_reg[M(i, 3)] = dst_reg[M(i, 3)] * r * c3;
    }
}
}  // namespace v7b

// ============================ dispatch ============================
template <uint32_t V>
void run(uint32_t eps_bits, uint32_t iters) {
    if constexpr (V == 0) {
        v0::sinkhorn(eps_bits, iters);
    } else if constexpr (V == 1) {
        v1::sinkhorn(eps_bits, iters);
    } else if constexpr (V == 2) {
        v2::sinkhorn(eps_bits, iters);
    } else if constexpr (V == 3) {
        v3::sinkhorn(eps_bits, iters);
    } else if constexpr (V == 4) {
        sinkhorn_v45<false>(eps_bits, iters);
    } else if constexpr (V == 5) {
        sinkhorn_v45<true>(eps_bits, iters);
    } else if constexpr (V == 6) {
        v6::sinkhorn(eps_bits, iters);
    } else if constexpr (V == 7) {
        v7b::sinkhorn(eps_bits, iters);
    } else if constexpr (V == 98) {
#pragma GCC unroll 32
        for (int k = 0; k < 32; ++k) {
            dst_reg[k] = static_cast<float>(k);
        }
    } else if constexpr (V == 99) {
        // nothing: copy + dispatch overhead only
    }
}

}  // namespace skb
