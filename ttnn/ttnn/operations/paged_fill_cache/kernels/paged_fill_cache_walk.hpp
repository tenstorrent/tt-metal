// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// The tile walk of paged_fill_cache, shared by the reader and the writer so the two can never diverge:
// both must visit the same mapped tiles in the same order (the k-th tile the reader puts into the ring is
// the k-th tile the writer takes out of it).
//
// A cluster owns the contiguous input pages [start, start + count). Input page p of [1, H, seq, D] is
// (h, t, d) with p = (h * seq_t + t) * D_t + d. Tile row t of head h goes to block
// b = row[t / bs_t]; when b == -1 the whole tile row is skipped (neither read nor written, rule R4);
// otherwise tile d of it goes to cache page ((b * H + h) * bs_t + t % bs_t) * D_t + d.
//
// The decomposition of `start` is done once; afterwards d -> t -> h advance as counters, and the
// page-table entry is looked up once per tile row entered (not per tile). The mapped tiles are dealt
// round-robin over the kernel's threads: thread `tid` of `num_threads` takes mapped tiles tid, tid + N, ...

#pragma once

#include <cstdint>

namespace paged_fill_cache {

constexpr int32_t SKIP_ENTRY = -1;

struct WalkShape {
    uint32_t num_heads;    // H
    uint32_t seq_tiles;    // seq_t
    uint32_t block_tiles;  // bs_t
    uint32_t d_tiles;      // D_t
};

// on_tile(input_page, cache_page) is called for every mapped tile of [start, start + count) that this
// thread owns, in input order.
template <typename OnTile>
inline void walk_mapped_tiles(
    const volatile tt_l1_ptr int32_t* row,
    const WalkShape& s,
    uint32_t start,
    uint32_t count,
    uint32_t tid,
    uint32_t num_threads,
    OnTile on_tile) {
    if (count == 0) {
        return;
    }
    uint32_t d = start % s.d_tiles;
    const uint32_t tile_row = start / s.d_tiles;
    uint32_t t = tile_row % s.seq_tiles;
    uint32_t h = tile_row / s.seq_tiles;
    uint32_t k = t / s.block_tiles;  // which block of the sequence: an index into the page-table row
    uint32_t r = t % s.block_tiles;  // which tile row of that block

    uint32_t page = start;
    uint32_t remaining = count;
    uint32_t turn = 0;  // the mapped-tile ordinal modulo num_threads
    while (remaining > 0) {
        const uint32_t run = (s.d_tiles - d) < remaining ? (s.d_tiles - d) : remaining;
        const int32_t block = row[k];
        if (block != SKIP_ENTRY) {
            const uint32_t dst = ((static_cast<uint32_t>(block) * s.num_heads + h) * s.block_tiles + r) * s.d_tiles + d;
            for (uint32_t i = 0; i < run; ++i) {
                if (turn == tid) {
                    on_tile(page + i, dst + i);
                }
                if (++turn == num_threads) {
                    turn = 0;
                }
            }
        }
        page += run;
        remaining -= run;
        d = 0;
        // Next tile row: t advances within the head, then the head.
        if (++r == s.block_tiles) {
            r = 0;
            ++k;
        }
        if (++t == s.seq_tiles) {
            t = 0;
            k = 0;
            r = 0;
            ++h;
        }
    }
}

}  // namespace paged_fill_cache
