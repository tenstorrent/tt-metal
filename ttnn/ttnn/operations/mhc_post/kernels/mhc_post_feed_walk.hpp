// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post feed (regime R4): the chunk walk shared by the port reader and the port sender, so both visit the same
// chunks in the same order.
//
// Chip `me` of a feed group of G sends column block d of its y (columns [d*Cb, (d+1)*Cb) tiles) to chip d, whose
// feed output D holds it at rows [me*T, (me+1)*T) of every image: D page j = Drow * Cb + col with
// Drow = img * G * Ht + me * Ht + tr (y row t = img * Ht + tr).
//
// The walk goes wave by wave: wave w covers the local rows [w*wave_rows, (w+1)*wave_rows) of every row group (the
// workers publish their rows in local-row order, all groups at once). Per destination, per row group, per image
// segment, the D pages of the segment form one contiguous index range [a, a + N); a chunk is up to run_pages pages
// a', a' + B, a' + 2B, ... of one DRAM bank (B = num_banks): consecutive in that bank, so one packet writes them.
// Banks are visited round-robin (fabric_all_gather's chunk walk). Chunk ci of a destination stream belongs to
// link ci % links.

#pragma once

#include <cstdint>

namespace mhc_feed {

struct Geom {
    uint32_t c_tiles;    // Ct: y column tiles
    uint32_t blk_tiles;  // Cb = Ct / G: column tiles of one destination block
    uint32_t ht;         // tile rows per image
    uint32_t g;          // G: chips in the feed group
    uint32_t me;         // my position in the group
    uint32_t r;          // R: row groups of the worker split
    uint32_t rows_base;  // rows of a group: rows_base (+1 for the first rows_extra groups)
    uint32_t rows_extra;
};

// first global tile row of row group `grp` and its row count
inline uint32_t group_t0(const Geom& g, uint32_t grp) {
    return grp * g.rows_base + (grp < g.rows_extra ? grp : g.rows_extra);
}
inline uint32_t group_rows(const Geom& g, uint32_t grp) { return g.rows_base + (grp < g.rows_extra ? 1 : 0); }

// y_output page (staging) of D page j sent by chip me to destination d
inline uint32_t src_page(const Geom& g, uint32_t d, uint32_t j) {
    const uint32_t drow = j / g.blk_tiles;
    const uint32_t col = j - drow * g.blk_tiles;
    const uint32_t img = drow / (g.g * g.ht);
    const uint32_t tr = drow - img * g.g * g.ht - g.me * g.ht;
    return (img * g.ht + tr) * g.c_tiles + d * g.blk_tiles + col;
}

// f(chunk index in the destination stream, first D page, pages) for every chunk of destination d in the local rows
// [i0, i1); `ci` carries the stream's chunk index across waves.
template <uint32_t num_banks, uint32_t run_pages, typename F>
inline void for_each_chunk(const Geom& g, uint32_t i0, uint32_t i1, uint32_t& ci, F&& f) {
    for (uint32_t grp = 0; grp < g.r; ++grp) {
        const uint32_t rows = group_rows(g, grp);
        const uint32_t lo = i0 < rows ? i0 : rows;
        const uint32_t hi = i1 < rows ? i1 : rows;
        uint32_t ta = group_t0(g, grp) + lo;
        const uint32_t tb = group_t0(g, grp) + hi;
        while (ta < tb) {
            const uint32_t img = ta / g.ht;
            const uint32_t seg_end = tb < (img + 1) * g.ht ? tb : (img + 1) * g.ht;
            const uint32_t a = ((img * g.g + g.me) * g.ht + (ta - img * g.ht)) * g.blk_tiles;
            const uint32_t end = a + (seg_end - ta) * g.blk_tiles;
            const uint32_t per_bank = (end - a + num_banks - 1) / num_banks;
            const uint32_t a_bank = a % num_banks;
            for (uint32_t m = 0; m < per_bank; m += run_pages) {
                for (uint32_t b = 0; b < num_banks; ++b) {
                    const uint32_t j0 = a + (b + num_banks - a_bank) % num_banks;  // first page of bank b
                    if (j0 >= end) {
                        continue;
                    }
                    const uint32_t cnt = (end - j0 + num_banks - 1) / num_banks;
                    if (m >= cnt) {
                        continue;
                    }
                    const uint32_t pages = (cnt - m) < run_pages ? (cnt - m) : run_pages;
                    f(ci++, j0 + m * num_banks, pages);
                }
            }
            ta = seg_end;
        }
    }
}

}  // namespace mhc_feed
