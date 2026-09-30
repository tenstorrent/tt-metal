// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post — shared segment / block derivation (included by reader, compute and writer).
//
// A core owns the contiguous flattened unit range [start_unit, start_unit + num_units), one unit being
// (token-tile row r, column tile c), r-major. A *segment* is the maximal run of those units inside one
// token-tile row; a segment is walked in blocks of `block_col_tiles` columns, the last one possibly
// ragged (`valid_col_tiles < block_col_tiles`). All three kernels iterate through THIS derivation so
// they cannot disagree on segment / block boundaries (a disagreement is a CB count mismatch -> hang).

#pragma once

#include <cstdint>

namespace mhc_post {

struct Segment {
    uint32_t row;        // token-tile row r
    uint32_t col0;       // first column tile c of the segment
    uint32_t col_tiles;  // number of column tiles in the segment
};

// Iterates the segments of one core's unit range.
struct SegmentWalker {
    uint32_t next_unit;
    uint32_t remaining;
    uint32_t col_tiles_per_row;  // Ct

    SegmentWalker(uint32_t start_unit, uint32_t num_units, uint32_t ct) :
        next_unit(start_unit), remaining(num_units), col_tiles_per_row(ct) {}

    bool done() const { return remaining == 0; }

    Segment next() {
        Segment s;
        s.row = next_unit / col_tiles_per_row;
        s.col0 = next_unit % col_tiles_per_row;
        const uint32_t row_left = col_tiles_per_row - s.col0;
        s.col_tiles = remaining < row_left ? remaining : row_left;
        next_unit += s.col_tiles;
        remaining -= s.col_tiles;
        return s;
    }
};

// Number of blocks of `block_col_tiles` covering `seg_col_tiles` (ceil).
inline uint32_t num_blocks(uint32_t seg_col_tiles, uint32_t block_col_tiles) {
    return (seg_col_tiles + block_col_tiles - 1) / block_col_tiles;
}

// Valid (runtime) column extent of block `block_idx` of a segment.
inline uint32_t block_valid_col_tiles(uint32_t seg_col_tiles, uint32_t block_col_tiles, uint32_t block_idx) {
    const uint32_t done = block_idx * block_col_tiles;
    const uint32_t left = seg_col_tiles - done;
    return left < block_col_tiles ? left : block_col_tiles;
}

// ---- core_balance: ramped block walk ----
// Block k of a core (counted across its segments) is min(B, R0 << k) columns wide (cut at segment ends), so every
// core's first request is small and the grid-wide start-up read burst shrinks. R0 = B is the op's walk exactly
// (blocks of B, ragged last block per segment). CB windows stay nominal (B); only `valid` varies.
struct RampBlk {
    uint32_t row, col_start, valid;
    bool first_of_segment;
};
template <uint32_t B, uint32_t R0>
struct RampBlocks {
    SegmentWalker w;
    Segment seg{0, 0, 0};
    uint32_t seg_done = 0;
    uint32_t k = 0;
    explicit RampBlocks(const SegmentWalker& walker) : w(walker) {}
    bool done() const { return seg_done >= seg.col_tiles && w.done(); }
    RampBlk next() {
        bool first = false;
        if (seg_done >= seg.col_tiles) {
            seg = w.next();
            seg_done = 0;
            first = true;
        }
        uint32_t width = B;
        if (R0 < B && k < 16) {
            const uint32_t r = R0 << k;
            width = r < B ? r : B;
        }
        const uint32_t left = seg.col_tiles - seg_done;
        const uint32_t valid = left < width ? left : width;
        RampBlk b{seg.row, seg.col0 + seg_done, valid, first};
        seg_done += valid;
        ++k;
        return b;
    }
};

// ---- core_balance: run-time claimed tail queues ----
// Queue v is the flattened unit range [qs, qe) with qe = end of core v's uniform-split range, qs = qe - q
// (pool_mode 2: one queue per core = its own tail, owner first, then stolen by others) or the single range
// [total - q, total) (pool_mode 1: one global pool, n1 = 1, g1 = total). A queue is cut into chunks aligned to
// `L` column boundaries inside each token row, also cut at the queue end, so a chunk never spans two rows and
// is at most one block. Claimed chunk (v, j) travels as tag = v * maxq + j + 1 (0 = end).
// Host mirror: candidate_descriptor._queue / _range_chunks.
struct PoolParams {
    uint32_t n1, g1, g2, q, maxq, ct, L;
    void queue(uint32_t v, uint32_t& qs, uint32_t& qe) const {
        const uint32_t start = v < n1 ? v * g1 : n1 * g1 + (v - n1) * g2;
        qe = start + (v < n1 ? g1 : g2);
        qs = qe - q;
    }
    uint32_t aligned_chunk(uint32_t unit) const {
        const uint32_t cpr = (ct + L - 1) / L;
        return (unit / ct) * cpr + (unit % ct) / L;
    }
    uint32_t queue_chunks(uint32_t v) const {
        uint32_t qs, qe;
        queue(v, qs, qe);
        return q == 0 ? 0 : aligned_chunk(qe - 1) - aligned_chunk(qs) + 1;
    }
    Segment chunk(uint32_t tag) const {
        const uint32_t v = (tag - 1) / maxq;
        const uint32_t j = (tag - 1) % maxq;
        uint32_t qs, qe;
        queue(v, qs, qe);
        const uint32_t cpr = (ct + L - 1) / L;
        const uint32_t g = aligned_chunk(qs) + j;
        Segment s;
        s.row = g / cpr;
        const uint32_t ca = (g % cpr) * L;
        s.col0 = j == 0 ? qs % ct : ca;
        const uint32_t end_col = ca + L < ct ? ca + L : ct;
        const uint32_t row_end = s.row * ct + end_col;
        const uint32_t u_end = row_end < qe ? row_end : qe;
        s.col_tiles = u_end - (s.row * ct + s.col0);
        return s;
    }
};

// ---- Expanded coefficient layout (cb_coef_bcast; reader writes it, compute reads it) ----
// Output stream j has n+1 terms: t = 0 is post_j, t = 1+i is comb[i][j] (comb applied transposed).
// Two terms share one fp32 tile: term t occupies HALF (t % 2) of tile j*P + t/2, with
// P = coef_tiles_per_stream = ceil((n+1)/2) (host CT arg). Half 0 = faces 0/2 (tile columns 0-15), half 1 =
// faces 1/3 (columns 16-31); every element of tile row rho in a half holds the coefficient of token row rho.
// A half carries everything a data face needs (the coefficient is constant along columns), so the SFPU
// reads data faces 2h and 2h+1 against coefficient face 2h + (t % 2).
constexpr uint32_t coef_tile_in_stream(uint32_t t) { return t / 2; }
constexpr uint32_t coef_half(uint32_t t) { return t % 2; }

}  // namespace mhc_post
