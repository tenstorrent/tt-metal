// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Shared by the compute-core reader, compute and writer kernels: the attention-unit decode. Units are ordered
// for (b, q head): [main q blocks], [joint q blocks] (head-major); core c owns units c, c + C, ..., c + (W-1)C, then at
// most one tail piece (see tail_piece) Every unit processes a nominal Sq-row block (CB credits are always nominal, so
// every block of every CB starts on the same slot); `rows` is the number of valid q tile-rows (reads and writes cover
// only those).
#pragma once

#include <cstdint>

struct AttnUnit {
    uint32_t bh;    // b * hq + h
    uint32_t row0;  // first q tile-row of the block within its tensor
    uint32_t rows;  // valid q tile-rows (<= Sq)
    bool joint;     // joint-Q unit (writes joint_out)
};

template <uint32_t Sq, uint32_t n_q_t, uint32_t Lq_t>
inline AttnUnit decode_unit(uint32_t u) {
    constexpr uint32_t nqb_main = (n_q_t + Sq - 1) / Sq;
    constexpr uint32_t nqb_joint = (Lq_t + Sq - 1) / Sq;
    constexpr uint32_t Uq = nqb_main + nqb_joint;
    AttnUnit a;
    a.bh = u / Uq;
    const uint32_t qb = u - a.bh * Uq;
    if (qb < nqb_main) {
        a.joint = false;
        a.row0 = qb * Sq;
        a.rows = (n_q_t - a.row0) < Sq ? (n_q_t - a.row0) : Sq;
    } else {
        a.joint = true;
        a.row0 = (qb - nqb_main) * Sq;
        a.rows = (Lq_t - a.row0) < Sq ? (Lq_t - a.row0) : Sq;
    }
    return a;
}

// Balanced tail split (host: plan_work_split): every core runs W = U / C whole units (c, c + C, ...), then at most
// one TAIL PIECE (the remainder units re-blocked into r-row pieces, r | Sq) read from 5 runtime args at `a`:
// has_tail, bh, row0, rows, joint. The piece is always the core's last unit.
inline AttnUnit tail_piece(uint32_t a) {
    AttnUnit t;
    t.bh = get_arg_val<uint32_t>(a + 1);
    t.row0 = get_arg_val<uint32_t>(a + 2);
    t.rows = get_arg_val<uint32_t>(a + 3);
    t.joint = get_arg_val<uint32_t>(a + 4) != 0;
    return t;
}

// ---------------------------------------------------------------------------------------------------------------
// Causal key-block plan (design regime R4), shared by the reader and the compute kernel so both walk the same blocks.
//
// Every device holds n_t tiles of the global sequence; tile t of device d sits at global tile position
//   contiguous: d * n_t + t
//   balanced (zig-zag, 2G chunks of n_t / 2 tiles): chunk d for t < n_t / 2, chunk 2G - 1 - d after it.
// Within one device the global position grows with t in both layouts, so for any query tile the visible keys of a
// source are a PREFIX of its tiles: a segment is truncated to the keys the unit's latest query can see (keys past
// that are never read nor computed; a source wholly in the future is skipped, landing wait included), and inside
// the kept blocks each (query tile, key tile) pair is full (gk < gq), diagonal (gk == gq: triangle mask) or dead
// (gk > gq: -inf mask). The local segment is walked first and its tile 0 (the device's earliest position) is
// visible to every query of the device, so the first processed block holds a valid key for every row.
// ---------------------------------------------------------------------------------------------------------------
template <bool balanced>
inline uint32_t global_tile(uint32_t dev, uint32_t t, uint32_t n_t, uint32_t G) {
    if constexpr (!balanced) {
        return dev * n_t + t;
    } else {
        const uint32_t h = n_t / 2;
        return t < h ? dev * h + t : (2 * G - 1 - dev) * h + (t - h);
    }
}

// Global tile position of compute row r (0-based in the unit) — rows past the unit's valid rows (nominal-credit
// padding rows of a ragged block) reuse the last valid row. Stacked heads (q_heads_per_unit) share positions.
template <bool balanced>
inline uint32_t unit_row_gtile(const AttnUnit& u, uint32_t r, uint32_t p, uint32_t n_q_head, uint32_t G) {
    const uint32_t rr = r < u.rows ? r : u.rows - 1;
    return global_tile<balanced>(p, (u.row0 + rr) % n_q_head, n_q_head, G);
}

template <bool balanced>
inline uint32_t unit_max_gtile(const AttnUnit& u, uint32_t p, uint32_t n_q_head, uint32_t G) {
    uint32_t m = 0;
    for (uint32_t r = 0; r < u.rows; ++r) {
        const uint32_t g = unit_row_gtile<balanced>(u, r, p, n_q_head, G);
        m = g > m ? g : m;
    }
    return m;
}

// Number of leading tiles (of `valid`) of source `src` with global position <= max_gq.
template <bool balanced>
inline uint32_t causal_visible_tiles(uint32_t valid, uint32_t src, uint32_t max_gq, uint32_t n_t, uint32_t G) {
    auto lead = [](int32_t first, int32_t m, int32_t cap) -> uint32_t {  // count of first + t <= m, t < cap
        const int32_t c = m - first + 1;
        return static_cast<uint32_t>(c < 0 ? 0 : (c > cap ? cap : c));
    };
    uint32_t n;
    if constexpr (!balanced) {
        n = lead(static_cast<int32_t>(src * n_t), static_cast<int32_t>(max_gq), static_cast<int32_t>(n_t));
    } else {
        const uint32_t h = n_t / 2;
        n = lead(static_cast<int32_t>(src * h), static_cast<int32_t>(max_gq), static_cast<int32_t>(h));
        if (n == h) {
            n += lead(
                static_cast<int32_t>((2 * G - 1 - src) * h), static_cast<int32_t>(max_gq), static_cast<int32_t>(h));
        }
    }
    return n < valid ? n : valid;
}

// Mask of one (query tile at gq, key tile at gk) pair: 0 = none, 1 = diagonal (mask tile 0), 2 = dead (mask tile 1).
inline uint32_t causal_mask_class(uint32_t gq, uint32_t gk) { return gk < gq ? 0 : (gk == gq ? 1 : 2); }
