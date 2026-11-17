// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_adaln — custom SFPU routines for the stats dev / publish pass, combine_block and normalize_block.
//
// Stats (cf_*): only COLUMN 0 of every stat tile is meaningful (row stats from REDUCE_ROW; bcast<Col> downstream reads
// column 0). SFPLOAD lane layout (see nz_load_cprime): column 0 lives in the even-parity vectors at SFPU rows
// {0, 2, 4, 6} of faces 0 and 2 (CF_O(0..7)), lanes 0/8/16/24. The publish / fold / finalize routines run once
// (VectorMode::None) over just those 8 vectors and use the same lanes of column 1 (+1, odd parity) and of faces 1/3
// (+CF_F1) as extra values per row: the publish emits [r_hi, r_lo | M2_hi, M2_lo] (fp32 DEST) / [p, M2 | d] (16-bit
// DEST), which the publish packs into compact vectors (one stat of all 32 rows per vector, see
// "compact_stats_payload" below); the fold / finalize run on compact vectors and expand m / rstd back to column 0.
// Slots are addressed absolutely from DEST tile 0 (CF_T SFPU rows per tile, CF_F1 = face 0 -> 1). The dev pass
// (ds_acc_strip) folds whole tiles into 4-row strip accumulators; ds_publish_compact lane-reduces them into the
// payload. 16-bit DEST: the stats values that are re-read from DEST (dev accumulators, fold hi parts, m, var + eps) are
// rounded to bf16 nearest in the LREG first (convert<vFloat16b>): stored unrounded they carried a downward bias into
// var (measured: 16-bit rel-RMS 0.0071 -> 0.0062 at 8192x256 wb, 0.0081 -> 0.0063 at 256x3072 wb). The normalize output
// is NOT pre-rounded (measured worse: 0.0032 -> 0.0048 at 256x3072 noaff, and slower).
//
// Raw-LLK deviation (helper gap): eltwise_chain cannot express (a) a fold over a CT-variable number of peers S whose
// accumulators stay live in DEST across peer copies, (b) a lane / face-offset operand/result (column 1, faces 1/3 as
// extra values per row), (c) a column-0-only binary SFPU op (binary *_tile APIs are VectorMode::RC only). Measured on
// BH (1,1,256,3072) dynamic_wb fp32: library-op sequence (baseline) 39.7 us; the same library ops on faces 0/2 with
// copy re-init hoisted 34.2 us; these fused routines 31.1 us (unpacked payload) / 28.7 us (packed payload).

#pragma once

#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_macros.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "ckernel_sfpu_sqrt.h"

namespace ckernel::sfpu {

constexpr int CF_T = 32;  // SFPU rows per tile
constexpr int CF_F1 = 8;  // face 0 -> face 1 (face 2 -> face 3) offset, SFPU rows
// v-th column-0 vector (v = 0..7): SFPU rows 0, 2, 4, 6 of face 0, then of face 2.
#define CF_O(v) (((v) & 3) * 2 + ((v) >> 2) * 16)

// fp32 payload split: x_hi = x with the low 13 mantissa bits cleared (exact in tf32), x_lo = x - x_hi (exact in fp32,
// <= 13 significant bits; srcA's tf32 keeps 11). The combine copies the payload through srcA (no unpack-to-dest
// handshake per tile) and re-forms x = x_hi + x_lo: relative error <= ~2^-21 (measured: bf16 outputs bit-identical
// to the fp32 unpack-to-dest payload, offsets 0 / 1000, std 1 / 100).
sfpi_inline sfpi::vFloat cf_hi(sfpi::vFloat x, sfpi::vInt mask) {
    return sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(x) & mask);
}

// Stats dev pass (stats_publish_fuse + dev_pass_macro): DEST-resident STRIP accumulators, one tile_regs_acquire per
// tile row. The row stats only need sum (x - p), sum (x - p)^2 per tile ROW. An SFPU vector (DEST rows 4t..4t+3, one
// column parity of one face) holds 4 rows x 8 columns, so the 4 vectors of a tile that cover the same 4 rows (faces
// 2fr / 2fr + 1, parity 0 / 1) fold into ONE accumulator vector: 8 s + 8 q vectors per row, packed into one DEST tile
//   s_g at CF_O(g) (faces 0/2, parity 0), q_g at CF_F1 + CF_O(g) (faces 1/3, parity 0), g = 0..7,
// instead of a whole s tile + q tile re-loaded / re-stored per window. Windows get one more work slot (fp32: 3,
// 16-bit: 7), per element only load + add + MAD remain, and two strips are interleaved so consecutive MADs are
// independent. Work tile j sits in DEST slot 1 + j; NJ work tiles this window. FIRST seeds the accumulators.
// MOVE_P (with FIRST): D0 holds the pivot tile on entry; its column-0 vectors CF_O(g) are moved to CF_O(g) + 1 (the
// parity-1 vectors the accumulators never use) before the accumulators overwrite CF_O(g). Run with VectorMode::None.
template <int NJ, bool FIRST, bool MOVE_P = false>
inline void ds_acc_strip(uint32_t, uint32_t, uint32_t) {
#pragma GCC unroll 4
    for (int gp = 0; gp < 4; gp++) {
        const int oA = CF_O(2 * gp);
        const int oB = CF_O(2 * gp + 1);
        sfpi::vFloat sA, qA, sB, qB;
        if constexpr (FIRST && MOVE_P) {
            sfpi::vFloat pA = sfpi::dst_reg[oA];
            sfpi::vFloat pB = sfpi::dst_reg[oB];
            sfpi::dst_reg[oA + 1] = pA;
            sfpi::dst_reg[oB + 1] = pB;
        }
        if constexpr (FIRST) {
            sfpi::vFloat vA = sfpi::dst_reg[CF_T + oA];
            sfpi::vFloat vB = sfpi::dst_reg[CF_T + oB];
            sA = vA;
            qA = vA * vA;
            sB = vB;
            qB = vB * vB;
        } else {
            sA = sfpi::dst_reg[oA];
            sB = sfpi::dst_reg[oB];
            qA = sfpi::dst_reg[CF_F1 + oA];
            qB = sfpi::dst_reg[CF_F1 + oB];
        }
#pragma GCC unroll 7
        for (int j = 0; j < NJ; j++) {
#pragma GCC unroll 4
            for (int e = (FIRST && j == 0) ? 1 : 0; e < 4; e++) {
                const int off = (1 + j) * CF_T + (e >> 1) * CF_F1 + (e & 1);
                sfpi::vFloat vA = sfpi::dst_reg[off + oA];
                sfpi::vFloat vB = sfpi::dst_reg[off + oB];
                sA += vA;
                sB += vB;
                qA += vA * vA;
                qB += vB * vB;
            }
        }
        if constexpr (!DST_ACCUM_MODE) {
            // A 16-bit DEST store truncates; round to nearest so the per-window re-store is unbiased.
            sA = sfpi::convert<sfpi::vFloat16b>(sA, sfpi::RoundMode::Nearest);
            sB = sfpi::convert<sfpi::vFloat16b>(sB, sfpi::RoundMode::Nearest);
            qA = sfpi::convert<sfpi::vFloat16b>(qA, sfpi::RoundMode::Nearest);
            qB = sfpi::convert<sfpi::vFloat16b>(qB, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[oA] = sA;
        sfpi::dst_reg[oB] = sB;
        sfpi::dst_reg[CF_F1 + oA] = qA;
        sfpi::dst_reg[CF_F1 + oB] = qB;
    }
}

// Sum of the 8 lanes of each SFPU subvector (= one DEST row's 8 same-parity columns), broadcast to all 8 lanes:
// rotate-and-add butterfly with SFPSHFT2 SUBVEC_SHFLROR1 (1 + 2 + 4 rotations, 3 adds, fp32).
sfpi_inline sfpi::vFloat ds_lane8_sum(sfpi::vFloat v) {
    sfpi::vFloat t = sfpi::subvec_shflror1(v);
    v = v + t;
    t = sfpi::subvec_shflror1(v);
    t = sfpi::subvec_shflror1(t);
    v = v + t;
    t = sfpi::subvec_shflror1(v);
    t = sfpi::subvec_shflror1(t);
    t = sfpi::subvec_shflror1(t);
    t = sfpi::subvec_shflror1(t);
    return v + t;
}

// ---- compact_stats_payload: the all-gather payload as a contiguous 1/8-tile prefix ----
// Compact layout: one SFPU vector (32 lanes = 4 subvector rows j x 8 lane-columns k) holds one stat for all 32 tile
// rows: lane (j, k) = tile row 4 * v + j with v = (8 - k) & 7, i.e. lane-column k carries the column-0 vector CF_O(v).
// A compact vector at unit u (u = 0..3) lies in face 0 rows 0..7, so the stats of up to 4 compact vectors are the
// first acc_tile_bytes / 8 bytes of the packed tile (fp32 512 B, bf16 256 B): the writer sends only that prefix.
// Built by a Horner insert (insert at lane-column 0, rotate right by one lane-column, 8 times; SFPSHFT2
// SUBVEC_SHFLROR1 moves lane-column k to k + 1) and expanded by the inverse rotations; moves are bitwise (AND / OR on
// the raw words), so every carried value is bit-exact. The fold / finalize run on the compact vectors (1 vector
// instead of 8 column-0 vectors), with the same arithmetic per row in the same order as the column-0 routines, so
// the results are bit-identical to them.
// Raw-LLK (helper gap): no helper / chain element moves SFPU lanes across lane-columns, and no pack helper emits a
// partial-tile payload; the exchange it shrinks is NoC-volume bound (writer exchange_streamed).

// All-ones in lane-column 0 (vConstTileId = 2 * lane), zero elsewhere.
sfpi_inline sfpi::vInt cf_col0_mask() {
    sfpi::vInt m = 0;
    v_if((sfpi::vConstTileId & 14) == 0) { m = -1; }
    v_endif;
    return m;
}

// Compact vector of the column-0 vectors at units BASE + CF_O(v), v = 0..7 (column 0 only, other lanes dropped).
template <int BASE>
sfpi_inline sfpi::vInt cf_gather(sfpi::vInt m) {
    sfpi::vInt acc = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[BASE + CF_O(1)])) & m;
#pragma GCC unroll 7
    for (int i = 2; i <= 8; i++) {
        const sfpi::vInt x = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[BASE + CF_O(i & 7)])) & m;
        sfpi::vFloat r = sfpi::subvec_shflror1(sfpi::as<sfpi::vFloat>(acc));
        acc = sfpi::as<sfpi::vInt>(r) | x;
    }
    return acc;
}

// Compact vector at unit SRC -> column 0 of the tile whose first unit is DST (other lanes carry other rows' values).
template <int SRC, int DST>
sfpi_inline void cf_scatter() {
    sfpi::vFloat a = sfpi::dst_reg[SRC];
#pragma GCC unroll 8
    for (int v = 0; v < 8; v++) {
        sfpi::dst_reg[DST + CF_O(v)] = a;
        if (v < 7) {
            a = sfpi::subvec_shflror1(a);
        }
    }
}

// Publish straight from the strip accumulators, compacted in the same pass (dev_pass_macro + compact_stats_payload,
// tournament 6; fused by stats_tail, Perf 1 of tournament 7, which deleted the two-pass ds_lane_publish +
// cf_compact_payload). In: D0 = strip accumulators, p at DEST vector P_OFF + CF_O(g). Per strip group g:
// d = sum / n, M2 = s2 - sum * d, r = p + d, with the row sums kept in fp32 LREGs (no DEST round trip). Out (D0 units
// 0..3, the 1/8-tile prefix the writer sends): fp32 DEST r_hi, r_lo, M2_hi, M2_lo; 16-bit DEST p, M2, d (stored
// truncated, as before: RNE of M2 / d measured worse, perf_experiments/dev_pass_macro/ab_rne16.jsonl). Each strip group
// g's stats are inserted straight into the compact vectors held in LREGs (Horner insert at lane-column 0, g visited 1,
// 2, .., 7, 0 as cf_gather does), so the 8 column-0 payload vectors per stat are never stored to DEST and re-loaded;
// the fp32 hi / lo split runs once on the compact vectors (elementwise, so the same bits). inv_n lives in
// vConstFloatPrgm0 (frees an LREG). Payload bits are identical to the two-pass version.
template <int P_OFF>
inline void ds_publish_compact(uint32_t, uint32_t, uint32_t, uint32_t inv_n_bits) {
    sfpi::vConstFloatPrgm0 = Converter::as_float(inv_n_bits);
    const sfpi::vInt m = cf_col0_mask();
    sfpi::vInt a0, a1;
    [[maybe_unused]] sfpi::vInt a2;
#pragma GCC unroll 8
    for (int i = 1; i <= 8; i++) {
        const int o = CF_O(i & 7);
        sfpi::vFloat sdev = ds_lane8_sum(sfpi::dst_reg[o]);
        sfpi::vFloat s2 = ds_lane8_sum(sfpi::dst_reg[CF_F1 + o]);
        sfpi::vFloat d = sdev * sfpi::vConstFloatPrgm0;
        sfpi::vFloat m2 = s2 - sdev * d;
        sfpi::vFloat p = sfpi::dst_reg[P_OFF + o];
        sfpi::vInt x0, x1;
        [[maybe_unused]] sfpi::vInt x2;
        if constexpr (DST_ACCUM_MODE) {
            x0 = sfpi::as<sfpi::vInt>(p + d) & m;
            x1 = sfpi::as<sfpi::vInt>(m2) & m;
        } else {
            x0 = sfpi::as<sfpi::vInt>(p) & m;
            x1 = sfpi::as<sfpi::vInt>(m2) & m;
            x2 = sfpi::as<sfpi::vInt>(d) & m;
        }
        if (i == 1) {
            a0 = x0;
            a1 = x1;
            if constexpr (!DST_ACCUM_MODE) {
                a2 = x2;
            }
        } else {
            a0 = sfpi::as<sfpi::vInt>(sfpi::subvec_shflror1(sfpi::as<sfpi::vFloat>(a0))) | x0;
            a1 = sfpi::as<sfpi::vInt>(sfpi::subvec_shflror1(sfpi::as<sfpi::vFloat>(a1))) | x1;
            if constexpr (!DST_ACCUM_MODE) {
                a2 = sfpi::as<sfpi::vInt>(sfpi::subvec_shflror1(sfpi::as<sfpi::vFloat>(a2))) | x2;
            }
        }
    }
    if constexpr (DST_ACCUM_MODE) {
        const sfpi::vInt hm = (int)0xFFFFE000;
        sfpi::vFloat r = sfpi::as<sfpi::vFloat>(a0);
        sfpi::vFloat rh = cf_hi(r, hm);
        sfpi::dst_reg[0] = rh;
        sfpi::dst_reg[1] = r - rh;
        sfpi::vFloat q = sfpi::as<sfpi::vFloat>(a1);
        sfpi::vFloat qh = cf_hi(q, hm);
        sfpi::dst_reg[2] = qh;
        sfpi::dst_reg[3] = q - qh;
    } else {
        sfpi::dst_reg[0] = sfpi::as<sfpi::vFloat>(a0);
        sfpi::dst_reg[1] = sfpi::as<sfpi::vFloat>(a1);
        sfpi::dst_reg[2] = sfpi::as<sfpi::vFloat>(a2);
    }
}

// Combine prologue: slot 1 (pivot tile, column 0 = p) -> compact p at slot 1 unit 0.
inline void cf_compact_pivot(uint32_t, uint32_t, uint32_t) {
    const sfpi::vInt m = cf_col0_mask();
    sfpi::dst_reg[CF_T] = sfpi::as<sfpi::vFloat>(cf_gather<CF_T>(m));
}

// fp32 compact fold of one peer: slot0 units 0 / 1 = sum d / sum q, slot1 unit 0 = p_own, slot2 = compact payload.
template <bool FIRST>
inline void cf_fold_fp32_c(uint32_t, uint32_t, uint32_t, uint32_t n_bits) {
    const sfpi::vFloat n = Converter::as_float(n_bits);
    const sfpi::vFloat p = sfpi::dst_reg[CF_T];
    sfpi::vFloat r = sfpi::dst_reg[2 * CF_T];
    r = r + sfpi::dst_reg[2 * CF_T + 1];
    sfpi::vFloat m2 = sfpi::dst_reg[2 * CF_T + 2];
    m2 = m2 + sfpi::dst_reg[2 * CF_T + 3];
    sfpi::vFloat d = r - p;
    sfpi::vFloat q = n * d * d + m2;
    if constexpr (!FIRST) {
        sfpi::vFloat sd0 = sfpi::dst_reg[0];
        sfpi::vFloat sq0 = sfpi::dst_reg[1];
        d = sd0 + d;
        q = sq0 + q;
    }
    sfpi::dst_reg[0] = d;
    sfpi::dst_reg[1] = q;
}

// 16-bit compact fold of one peer (streamed: one call per landed payload): delta = (p_j - p_own) + d_j; sum d +=
// delta, sum q += M2_j + n delta^2. slot0 units 0 / 1 = sum d hi / lo, units 2 / 3 = sum q hi / lo, slot1 unit 0 =
// p_own, slot2 units 0..2 = [p, M2, d]. The running sums cross calls in a 16-bit DEST, so they are kept compensated
// (hi = bf16-rounded in the LREG, lo = exact - hi): ~16 mantissa bits instead of 8.
template <bool FIRST>
inline void cf_fold_b16_c(uint32_t, uint32_t, uint32_t, uint32_t n_bits) {
    const sfpi::vFloat n = Converter::as_float(n_bits);
    const sfpi::vFloat p = sfpi::dst_reg[CF_T];
    sfpi::vFloat pj = sfpi::dst_reg[2 * CF_T];
    sfpi::vFloat m2j = sfpi::dst_reg[2 * CF_T + 1];
    sfpi::vFloat dj = sfpi::dst_reg[2 * CF_T + 2];
    sfpi::vFloat sd = (pj - p) + dj;
    sfpi::vFloat sq = n * sd * sd + m2j;
    if constexpr (!FIRST) {
        sfpi::vFloat sd0 = sfpi::dst_reg[0];
        sd0 = sd0 + sfpi::dst_reg[1];
        sfpi::vFloat sq0 = sfpi::dst_reg[2];
        sq0 = sq0 + sfpi::dst_reg[3];
        sd = sd0 + sd;
        sq = sq0 + sq;
    }
    sfpi::vFloat sdh = sfpi::convert<sfpi::vFloat16b>(sd, sfpi::RoundMode::Nearest);
    sfpi::vFloat sqh = sfpi::convert<sfpi::vFloat16b>(sq, sfpi::RoundMode::Nearest);
    sfpi::dst_reg[0] = sdh;
    sfpi::dst_reg[2] = sqh;
    sfpi::dst_reg[1] = sd - sdh;
    sfpi::dst_reg[3] = sq - sqh;
}

// wide_groups ragged slices: weighted folds (n = this slot's slice element count). Accumulate sum n delta and
// sum M2 + n delta^2; the finalize runs with n = 1, inv_s = 1 / W (m = sum n delta / W, M2 = sq - sd * m).
template <bool FIRST>
inline void cf_fold_fp32_w(uint32_t, uint32_t, uint32_t, uint32_t n_bits) {
    const sfpi::vFloat n = Converter::as_float(n_bits);
    const sfpi::vFloat p = sfpi::dst_reg[CF_T];
    sfpi::vFloat r = sfpi::dst_reg[2 * CF_T];
    r = r + sfpi::dst_reg[2 * CF_T + 1];
    sfpi::vFloat m2 = sfpi::dst_reg[2 * CF_T + 2];
    m2 = m2 + sfpi::dst_reg[2 * CF_T + 3];
    sfpi::vFloat d = r - p;
    sfpi::vFloat nd = n * d;
    sfpi::vFloat q = nd * d + m2;
    if constexpr (!FIRST) {
        sfpi::vFloat sd0 = sfpi::dst_reg[0];
        sfpi::vFloat sq0 = sfpi::dst_reg[1];
        nd = sd0 + nd;
        q = sq0 + q;
    }
    sfpi::dst_reg[0] = nd;
    sfpi::dst_reg[1] = q;
}

template <bool FIRST>
inline void cf_fold_b16_w(uint32_t, uint32_t, uint32_t, uint32_t n_bits) {
    const sfpi::vFloat n = Converter::as_float(n_bits);
    const sfpi::vFloat p = sfpi::dst_reg[CF_T];
    sfpi::vFloat pj = sfpi::dst_reg[2 * CF_T];
    sfpi::vFloat m2j = sfpi::dst_reg[2 * CF_T + 1];
    sfpi::vFloat dj = sfpi::dst_reg[2 * CF_T + 2];
    sfpi::vFloat dl = (pj - p) + dj;
    sfpi::vFloat sd = n * dl;
    sfpi::vFloat sq = sd * dl + m2j;
    if constexpr (!FIRST) {
        sfpi::vFloat sd0 = sfpi::dst_reg[0];
        sd0 = sd0 + sfpi::dst_reg[1];
        sfpi::vFloat sq0 = sfpi::dst_reg[2];
        sq0 = sq0 + sfpi::dst_reg[3];
        sd = sd0 + sd;
        sq = sq0 + sq;
    }
    sfpi::vFloat sdh = sfpi::convert<sfpi::vFloat16b>(sd, sfpi::RoundMode::Nearest);
    sfpi::vFloat sqh = sfpi::convert<sfpi::vFloat16b>(sq, sfpi::RoundMode::Nearest);
    sfpi::dst_reg[0] = sdh;
    sfpi::dst_reg[2] = sqh;
    sfpi::dst_reg[1] = sd - sdh;
    sfpi::dst_reg[3] = sq - sqh;
}

// rsqrt in place at unit A, one vector: rsqrt_tile's non-approximate body and constants (sqrt_init<false>), rounded
// to bf16 in 16-bit DEST as rsqrt_tile does (replaces rsqrt_tile_init + rsqrt_tile(VectorMode::C): bit-identical).
template <int A>
sfpi_inline void cf_rsqrt_at() {
    sfpi::vConstIntPrgm0 = 0x5f1110a0;
    sfpi::vConstFloatPrgm1 = 2.2825186f;
    sfpi::vConstFloatPrgm2 = 2.2533049f;
    sfpi::vFloat y = _calculate_sqrt_body_<false, true, false>(sfpi::dst_reg[A]);
    if constexpr (!DST_ACCUM_MODE) {
        y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
    }
    sfpi::dst_reg[A] = y;
}

// Compact finalize: m -> slot0 unit 8, var + eps -> slot1 unit 8, rsqrt there, then both expanded to column 0 of
// slots 0 / 1 (the layout cb_row_stats' Col-bcast consumers read).
inline void cf_finalize_c(
    uint32_t, uint32_t, uint32_t, uint32_t n_bits, uint32_t inv_s_bits, uint32_t inv_w_bits, uint32_t eps_bits) {
    const sfpi::vFloat n = Converter::as_float(n_bits);
    const sfpi::vFloat inv_s = Converter::as_float(inv_s_bits);
    const sfpi::vFloat inv_w = Converter::as_float(inv_w_bits);
    const sfpi::vFloat eps = Converter::as_float(eps_bits);
    sfpi::vFloat sd, sq;
    if constexpr (DST_ACCUM_MODE) {
        sd = sfpi::dst_reg[0];
        sq = sfpi::dst_reg[1];
    } else {
        sd = sfpi::dst_reg[0];
        sd = sd + sfpi::dst_reg[1];
        sq = sfpi::dst_reg[2];
        sq = sq + sfpi::dst_reg[3];
    }
    sfpi::vFloat m = sd * inv_s;
    sfpi::vFloat m2 = sq - n * sd * m;
    v_if(m2 < 0.0f) { m2 = 0.0f; }
    v_endif;
    if constexpr (DST_ACCUM_MODE) {
        sfpi::dst_reg[CF_F1] = m;
        sfpi::dst_reg[CF_T + CF_F1] = m2 * inv_w + eps;
    } else {
        sfpi::dst_reg[CF_F1] = sfpi::convert<sfpi::vFloat16b>(m, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[CF_T + CF_F1] = sfpi::convert<sfpi::vFloat16b>(m2 * inv_w + eps, sfpi::RoundMode::Nearest);
    }
    cf_rsqrt_at<CF_T + CF_F1>();
    cf_scatter<CF_F1, 0>();
    cf_scatter<CF_T + CF_F1, CF_T>();
}

// ---- normalize (fp32 DEST): c'_k = -(p + m) * rstd preloaded into DEST, then one FPU MAC D += x * rstd ----

// c' = -(p + m) * rstd from slot 0 = p, slot 1 = m, slot 2 = rstd (Col-broadcast tiles), only at the 8 strip vectors
// nz_load_cprime reads (face 0 / face 2, parity 0; unit 16 * (L / 4) + 2 * (L % 4)), written back to slot 0. Same
// expression as the former all-32-vector nz_cprime -> bit-identical c'. Caller has done _llk_math_eltwise_sfpu_start_.
inline void nz_cprime_strips() {
#pragma GCC unroll 8
    for (int L = 0; L < 8; L++) {
        const int unit = 16 * (L / 4) + 2 * (L % 4);
        sfpi::vFloat p = sfpi::dst_reg[unit];
        sfpi::vFloat m = sfpi::dst_reg[CF_T + unit];
        sfpi::vFloat r = sfpi::dst_reg[2 * CF_T + unit];
        sfpi::dst_reg[unit] = -((p + m) * r);
    }
}

// c' stays in LREG0..7 for a whole tile row: one vector per 4-row strip (L = 4 * face_row + strip). SFPLOAD lane
// layout: row = (addr & ~3) + lane / 8, col = 2 * (lane & 7) + ((addr & 2) != 0), so every lane of a strip vector
// holds that row's c' (c' is constant along a row) and one vector serves both face columns and both column parities.
// INT32 load / store = raw copy of the fp32 DEST word (no rounding). Nothing else in normalize_block touches the
// SFPU, so the LREGs survive the FPU ops and DEST-half swaps between windows. ADDR_MOD_7 = zero increment (SFPU init).
#define NZ_LD(L, A) TTI_SFPLOAD(L, InstrModLoadStore::INT32, ADDR_MOD_7, A)
#define NZ_ST(L, A) TTI_SFPSTORE(L, InstrModLoadStore::INT32, ADDR_MOD_7, A)
// From the c' tile at the current DEST address: face 0 strips (rows 0..15) -> L0..3, face 2 (rows 16..31) -> L4..7.
inline void nz_load_cprime() {
    NZ_LD(0, 0);
    NZ_LD(1, 4);
    NZ_LD(2, 8);
    NZ_LD(3, 12);
    NZ_LD(4, 32);
    NZ_LD(5, 36);
    NZ_LD(6, 40);
    NZ_LD(7, 44);
}
// Write c' into the whole tile at the current DEST address: face f at +16 f, strip t at +4 t, parity q at +2 q.
#define NZ_ST_STRIP(L, FR, T)       \
    NZ_ST(L, 32 * FR + 4 * T);      \
    NZ_ST(L, 32 * FR + 4 * T + 2);  \
    NZ_ST(L, 32 * FR + 16 + 4 * T); \
    NZ_ST(L, 32 * FR + 16 + 4 * T + 2);
inline void nz_store_cprime_tile() {
    NZ_ST_STRIP(0, 0, 0)
    NZ_ST_STRIP(1, 0, 1)
    NZ_ST_STRIP(2, 0, 2)
    NZ_ST_STRIP(3, 0, 3)
    NZ_ST_STRIP(4, 1, 0)
    NZ_ST_STRIP(5, 1, 1)
    NZ_ST_STRIP(6, 1, 2)
    NZ_ST_STRIP(7, 1, 3)
}

// ---- sfpu_normalize: affine tail on the SFPU (fp32 DEST: D = D * w + b in fp32 LREGs) ----
// P ("wb prep") tile, bf16 in L1, built once per call: G = 3 column tiles per P tile, column g in face g:
//   unit 8g + 2h + q     = w (column half h, parity q), every row of the 4-row strip equal (row-replicated)
//   unit 8g + 4 + 2h + q = b (same)
//   face 3 (units 24..31) = zero; per-window c' stash (fp32 DEST) / row stats S (16-bit DEST, one FPU add).
// SFPLOAD lane layout (see nz_load_cprime): one vector at (face, strip t, parity q) covers rows 4t..4t+3 x the 8
// columns of parity q of that face; a row-replicated w strip therefore matches every strip of the x face.
constexpr int NZ_G = 3;

// slot0 = w Row-bcast tile, slot1 = b Row-bcast tile -> slot2 face G (one column).
template <int G>
inline void nz_wbp_prep() {
#pragma GCC unroll 2
    for (int h = 0; h < 2; h++) {
#pragma GCC unroll 2
        for (int q = 0; q < 2; q++) {
            sfpi::vFloat w = sfpi::dst_reg[8 * h + q];
            sfpi::vFloat b = sfpi::dst_reg[CF_T + 8 * h + q];
            sfpi::dst_reg[2 * CF_T + 8 * G + 2 * h + q] = w;
            sfpi::dst_reg[2 * CF_T + 8 * G + 4 + 2 * h + q] = b;
        }
    }
}

// c' stash (fp32 DEST, window of 3 + P in slot 3): face 3 of the P tile, unit 24 + L -> addr 3*64 + 48 + 2L.
inline void nz_stash_cprime() {
    NZ_ST(0, 240);
    NZ_ST(1, 242);
    NZ_ST(2, 244);
    NZ_ST(3, 246);
    NZ_ST(4, 248);
    NZ_ST(5, 250);
    NZ_ST(6, 252);
    NZ_ST(7, 254);
}
inline void nz_unstash_cprime() {
    NZ_LD(0, 240);
    NZ_LD(1, 242);
    NZ_LD(2, 244);
    NZ_LD(3, 246);
    NZ_LD(4, 248);
    NZ_LD(5, 250);
    NZ_LD(6, 252);
    NZ_LD(7, 254);
}

// ---- sfpu_normalize full16 (16-bit DEST): D = x - p (FPU), then y = (D * rstd + c'') * w + b on the SFPU, c'' =
// -m*rstd. Per-row stats tiles (bf16, built once per row): S_r = rstd strips, S_c = c'' strips, each in face 3 (unit 24
// + L, L = 4 * face_row + strip), every other position zero, so one FPU add P + S lands both in one DEST slot. Zero
// face 3 of slot 2 (the P tile being built).
inline void nz_wbp_zero_face3() {
#pragma GCC unroll 8
    for (int u = 0; u < 8; u++) {
        sfpi::dst_reg[2 * CF_T + 24 + u] = 0.0f;
    }
}
// slot0 = m Col-bcast, slot1 = rstd Col-bcast -> slot2 = S_r, slot3 = S_c.
inline void nz_stats16_build() {
    // faces 0..2 of slots 2 / 3 are already zero: the pack's release clears each DEST half (ZEROACC CLR_HALF).
#pragma GCC unroll 8
    for (int L = 0; L < 8; L++) {
        const int unit = 16 * (L / 4) + 2 * (L % 4);
        sfpi::vFloat m = sfpi::dst_reg[unit];
        sfpi::vFloat rs = sfpi::dst_reg[CF_T + unit];
        sfpi::dst_reg[2 * CF_T + 24 + L] = rs;
        sfpi::dst_reg[3 * CF_T + 24 + L] = -(m * rs);
    }
}

// noaff full16: one stats tile S = [0 | 0 | rstd strips (face 2, unit 16 + L) | c'' strips (face 3, unit 24 + L)].
// slot0 = m Col-bcast, slot1 = rstd Col-bcast -> slot2 = S.
inline void nz_stats16_build_noaff() {
    // faces 0 / 1 of slot 2 are already zero (the pack's release clears each DEST half).
#pragma GCC unroll 8
    for (int L = 0; L < 8; L++) {
        const int unit = 16 * (L / 4) + 2 * (L % 4);
        sfpi::vFloat m = sfpi::dst_reg[unit];
        sfpi::vFloat rs = sfpi::dst_reg[CF_T + unit];
        sfpi::dst_reg[2 * CF_T + 16 + L] = rs;
        sfpi::dst_reg[2 * CF_T + 24 + L] = -(m * rs);
    }
}

// ---- normalize_sfpu_macro: SFPLOADMACRO-pipelined normalize MADs (perf_experiments/normalize_sfpu_macro) ----
// One SFPLOADMACRO per DEST vector does load -> MAD -> store, so the SFPU retires one vector per instruction slot
// instead of the ~3+ slots of an explicit SFPLOAD / SFPMAD / SFPSTORE sequence. Same fp32 MAD (one rounding, same
// operands, same order), so the result is bit-identical to the sfpi routines it replaces.
// Two instruction templates, both y = A * x + C with x = the macro-loaded vector (VB := VD):
//   template 0: A = LREG4, C = LREG5  (macro 0)      template 1: A = LREG6, C = LREG7  (macro 1)
// The operand pair alternates between the two sets per operand change, so an explicit reload of one set never
// races a scheduled MAD still reading it. Loaded vectors rotate over LREG0..3.
// Schedule (delays in elapsed SFPU instructions): load at slot t, MAD at t + 1, store at t + 3 (MAD latency 2).
// The store goes back to the address the macro loaded from (same Mod0 as the load).
#define NM_MAC(M, V, A) TTI_SFPLOADMACRO(((M) << 2) | ((V) & 3), 0, ADDR_MOD_7, (A))
#define NM_LD(L, A) TTI_SFPLOAD(L, 0, ADDR_MOD_7, A)

inline void nm_macro_init() {
    TTI_SFPMAD(p_sfpu::LREG4, 0, p_sfpu::LREG5, 12, 0);  // InstructionTemplate[0]
    TTI_SFPMAD(p_sfpu::LREG6, 0, p_sfpu::LREG7, 13, 0);  // InstructionTemplate[1]
    constexpr uint32_t disabled = 7 << 3;  // a disabled unit uses delay 7 (cannot cancel a pending instruction)
    constexpr uint32_t store_bits = (2 << 3) | 3;
    {
        constexpr uint32_t mad_bits = 0x80 | (0 << 3) | (4 + 0);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | disabled);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | disabled);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }
    {
        constexpr uint32_t mad_bits = 0x80 | (0 << 3) | (4 + 1);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | disabled);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | disabled);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }
    {
        // Macro 2 (16-bit affine): template 0 at once, store 4 instructions later (after the explicit second MAD).
        constexpr uint32_t mad_bits = 0x80 | (0 << 3) | (4 + 0);
        constexpr uint32_t store4_bits = (4 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | disabled);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store4_bits << 8) | disabled);
        TTI_SFPCONFIG(0, 4 + 2, 0);
    }
    // Misc: StoreMod0 = 0, UsesLoadMod0ForStore = all, UnitDelayKind = elapsed instructions (all units).
    TTI_SFPCONFIG(0xff0, 8, 1);
}

// fp32 affine, x-slot J, P slot PS face G: per (column half h, parity q) one w / b pair (raw DEST addresses: tile 64,
// face 16, strip 4, parity 2) then the 8 x vectors (face rows r = 0, 1 x strips 0..3) through the macro of that set.
template <int J, int PS, int G>
inline void nm_affine_tile() {
#pragma GCC unroll 2
    for (int h = 0; h < 2; h++) {
#pragma GCC unroll 2
        for (int q = 0; q < 2; q++) {
            const int set = q;  // combos alternate q fastest -> sets alternate
            const int wa = 64 * PS + 16 * G + 4 * h + 2 * q;
            if (set == 0) {
                NM_LD(p_sfpu::LREG4, wa);
                NM_LD(p_sfpu::LREG5, wa + 8);
            } else {
                NM_LD(p_sfpu::LREG6, wa);
                NM_LD(p_sfpu::LREG7, wa + 8);
            }
#pragma GCC unroll 2
            for (int r = 0; r < 2; r++) {
#pragma GCC unroll 4
                for (int t = 0; t < 4; t++) {
                    const int a = 64 * J + 16 * (2 * r + h) + 4 * t + 2 * q;
                    NM_MAC(set, t, a);
                }
            }
        }
    }
}
template <int J, int NX>
inline void nm_affine_j() {
    nm_affine_tile<J, NX + J / NZ_G, J % NZ_G>();
}
template <int NX>
inline void nm_affine_window(uint32_t n) {
    nm_affine_j<0, NX>();
    if (n > 1) {
        nm_affine_j<1, NX>();
    }
    if (n > 2) {
        nm_affine_j<2, NX>();
    }
    TTI_SFPNOP;  // drain the last macros' scheduled MAD / store
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// 16-bit no-affine: y = D * rstd + c'' (S tile in slot SS: rstd face 2 unit 16 + L, c'' face 3 unit 24 + L).
// Strip-outer over the window: one rstd / c'' pair per strip L (4 x vectors per tile per strip).
template <int SS, int L>
inline void nm_norm16_strip(uint32_t n) {
    constexpr int set = L & 1;
    constexpr int ra = 64 * SS + 32 + 2 * L;
    constexpr int r = L / 4, t = L % 4;
    if constexpr (set == 0) {
        NM_LD(p_sfpu::LREG4, ra);
        NM_LD(p_sfpu::LREG5, ra + 16);
    } else {
        NM_LD(p_sfpu::LREG6, ra);
        NM_LD(p_sfpu::LREG7, ra + 16);
    }
#pragma GCC unroll 8
    for (int j = 0; j < 7; j++) {
        if (j > 0 && (uint32_t)j >= n) {
            break;
        }
        const int a = 64 * j + 32 * r + 4 * t;
        NM_MAC(set, 0, a);
        NM_MAC(set, 1, a + 2);
        NM_MAC(set, 2, a + 16);
        NM_MAC(set, 3, a + 18);
    }
}
template <int SS>
inline void nm_norm16_window(uint32_t n) {
    nm_norm16_strip<SS, 0>(n);
    nm_norm16_strip<SS, 1>(n);
    nm_norm16_strip<SS, 2>(n);
    nm_norm16_strip<SS, 3>(n);
    nm_norm16_strip<SS, 4>(n);
    nm_norm16_strip<SS, 5>(n);
    nm_norm16_strip<SS, 6>(n);
    nm_norm16_strip<SS, 7>(n);
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// 16-bit affine: y = (D * rstd + c'') * w + b. P tiles in slots 6, 7 (w / b, faces 0..2), S_r = rstd in slot 6 face 3,
// S_c = c'' in slot 7 face 3 (unit 24 + L). Strip-outer over the window (rstd / c'' of strip L in LREG4 / LREG5 for
// template 0); per x vector: macro 2 [load D; t = rstd * D + c''], then w / b into LREG6 / LREG7 and the explicit
// second MAD y = t * w + b; the macro's store fires 4 instructions after it. Slots per vector: M, LD w, LD b, X:
//   template MAD at M + 1 (result M + 3), X at M + 3 (result M + 5), store at M + 5; the MAD unit sees M + 1 and
//   M + 3 only, so the two MAD streams never collide. Same two MADs in the same order as the explicit sfpi routine it
//   replaced.
template <int L>
inline void nm_full16_strip(uint32_t n) {
    constexpr int r = L / 4, t = L % 4;
    NM_LD(p_sfpu::LREG4, 64 * 6 + 48 + 2 * L);
    NM_LD(p_sfpu::LREG5, 64 * 7 + 48 + 2 * L);
#pragma GCC unroll 6
    for (int j = 0; j < 6; j++) {
        if (j > 0 && (uint32_t)j >= n) {
            break;
        }
        const int ps = 6 + j / NZ_G, g = j % NZ_G;
#pragma GCC unroll 2
        for (int h = 0; h < 2; h++) {
#pragma GCC unroll 2
            for (int q = 0; q < 2; q++) {
                const int v = 2 * h + q;
                const int a = 64 * j + 16 * (2 * r + h) + 4 * t + 2 * q;
                const int wa = 64 * ps + 16 * g + 4 * h + 2 * q;
                NM_MAC(2, v, a);
                NM_LD(p_sfpu::LREG6, wa);
                NM_LD(p_sfpu::LREG7, wa + 8);
                TTI_SFPMAD(v, p_sfpu::LREG6, p_sfpu::LREG7, v, 0);
            }
        }
    }
}
inline void nm_full16_window(uint32_t n) {
    nm_full16_strip<0>(n);
    nm_full16_strip<1>(n);
    nm_full16_strip<2>(n);
    nm_full16_strip<3>(n);
    nm_full16_strip<4>(n);
    nm_full16_strip<5>(n);
    nm_full16_strip<6>(n);
    nm_full16_strip<7>(n);
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

}  // namespace ckernel::sfpu
#endif  // TRISC_MATH
