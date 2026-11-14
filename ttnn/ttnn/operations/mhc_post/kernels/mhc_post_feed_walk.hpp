// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post feed (regime R4): the walk shared by the port reader, the port sender and the landing merge, so all of
// them visit the same items in the same order.
//
// Chip `me` of a feed group of G sends column block d of its y (T logical rows a image) to chip d, whose feed
// output D holds it at D rows [me*T, (me+1)*T) of every image. D has htd = ceil(G*T / 32) tile rows a image, y has
// ht = ceil(T / 32). With T % 32 != 0 chip me's rows start at row offset (me*T) mod 32 of a D tile (Refinement 2b):
//   alone D tile rows   every valid row of the tile is mine: the port reader assembles each D tile from the (at most
//                       two) y tiles its rows span — whole 16-element face rows, 32 B, copied in runs over the NoC
//                       inside L1 — and the sender writes whole D pages (bank runs, as before);
//   shared D tile rows  the tile holds rows of several chips: never written by two chips (partial DRAM tile writes
//                       from different chips race at burst granularity). Each contributor sends ITS rows of the
//                       whole tile row (all Cb columns) to the landing on the destination's landing port in a
//                       compact layout; the landing port merges the full tiles and writes them to its own D.
// T % 32 == 0 has no shared row and every shift is 0: the walk is the plain bank-run walk of whole D pages.
//
// Compact landing of shared tile row s (Cb * 2048 B, s = img * nsh + rank of the row among the image's shared rows,
// the same on every chip): contributor rows [a, b) (rows of the 32-row tile) occupy [a*64*Cb, b*64*Cb); column c
// of it sits at c*(b-a)*64 and holds face 0 / 1 of its top rows, then face 2 / 3 of its bottom rows.
//
// Waves: item wave = the latest (local row index / wave_rows) of the y tile rows it reads; the reader waits until
// every worker published that wave. Page chunks: per destination, maximal runs of consecutive alone rows of one
// wave form a D page range [a, end); a chunk is up to run_pages pages of one DRAM bank (fabric_all_gather's chunk
// walk). Landing pieces: up to run_pages columns of one contributor block. Item ci of a (wave, destination) belongs
// to link ci % links.

#pragma once

#include <cstdint>

namespace mhc_feed {

constexpr uint32_t ROW_BYTES = 32;  // one bf16 face row: 16 elements
constexpr uint32_t FACE_BYTES = 512;
constexpr uint32_t TILE_ROW_BYTES = 2 * ROW_BYTES;  // one tile row over both column faces

struct Geom {
    uint32_t c_tiles;    // Ct: y column tiles
    uint32_t blk_tiles;  // Cb = Ct / G: column tiles of one destination block
    uint32_t ht;         // y tile rows per image
    uint32_t g;          // G: chips in the feed group
    uint32_t me;         // my position in the group
    uint32_t r;          // R: row groups of the worker split
    uint32_t rows_base;  // rows of a group: rows_base (+1 for the first rows_extra groups)
    uint32_t rows_extra;
    uint32_t t_rows;  // T: logical rows per image
    uint32_t htd;     // D tile rows per image: ceil(G * T / 32)
    uint32_t nimg;    // images
    uint32_t nsh;     // shared D tile rows per image
};

inline uint32_t min_u(uint32_t a, uint32_t b) { return a < b ? a : b; }
inline uint32_t max_u(uint32_t a, uint32_t b) { return a > b ? a : b; }

// first global tile row of row group `grp` and its row count
inline uint32_t group_t0(const Geom& g, uint32_t grp) {
    return grp * g.rows_base + (grp < g.rows_extra ? grp : g.rows_extra);
}
inline uint32_t group_rows(const Geom& g, uint32_t grp) { return g.rows_base + (grp < g.rows_extra ? 1 : 0); }
// local row index (inside its row group) of global y tile row t
inline uint32_t local_row(const Geom& g, uint32_t t) {
    const uint32_t big = g.rows_extra * (g.rows_base + 1);
    const uint32_t grp = t < big ? t / (g.rows_base + 1) : g.rows_extra + (t - big) / g.rows_base;
    return t - group_t0(g, grp);
}

// rows of chip q inside D tile row R (image-local): [a, b), empty when a >= b
inline uint32_t rows_a(const Geom& g, uint32_t q, uint32_t R) {
    const uint32_t lo = q * g.t_rows;
    return lo > 32 * R ? lo - 32 * R : 0;
}
inline uint32_t rows_b(const Geom& g, uint32_t q, uint32_t R) {
    const uint32_t hi = (q + 1) * g.t_rows;
    return hi <= 32 * R ? 0 : min_u(32, hi - 32 * R);
}
// first and last chip with rows in D tile row R
inline uint32_t first_chip(const Geom& g, uint32_t R) { return (32 * R) / g.t_rows; }
inline uint32_t last_chip(const Geom& g, uint32_t R) { return (min_u(32 * R + 32, g.g * g.t_rows) - 1) / g.t_rows; }
inline bool shared_row(const Geom& g, uint32_t R) { return first_chip(g, R) != last_chip(g, R); }
// rank of shared row R among the image's shared rows
inline uint32_t shared_rank(const Geom& g, uint32_t R) {
    uint32_t s = 0;
    for (uint32_t k = 0; k < R; ++k) {
        s += shared_row(g, k) ? 1 : 0;
    }
    return s;
}
// landing pieces one contributor sends for one shared tile row
template <uint32_t run_pages>
inline uint32_t pieces_per_block(const Geom& g) {
    return (g.blk_tiles + run_pages - 1) / run_pages;
}
// landing pieces the landing port receives a call: every contributor of every shared row of every image
template <uint32_t run_pages>
inline uint32_t landing_pieces(const Geom& g) {
    uint32_t n = 0;
    for (uint32_t R = 0; R < g.htd; ++R) {
        if (shared_row(g, R)) {
            n += last_chip(g, R) - first_chip(g, R) + 1;
        }
    }
    return n * g.nimg * pieces_per_block<run_pages>(g);
}

// L1 address of tile row k, column face h of a tile in face layout
inline uint32_t tile_row_addr(uint32_t base, uint32_t k, uint32_t h) {
    return base + (((k >> 4) << 1) + h) * FACE_BYTES + (k & 15) * ROW_BYTES;
}
// L1 address of row k (a <= k < b), face h of one column of a compact contributor block at `col`
inline uint32_t compact_row_addr(uint32_t col, uint32_t a, uint32_t b, uint32_t k, uint32_t h) {
    const uint32_t nt = a < 16 ? min_u(b, 16) - a : 0;
    if (k < 16) {
        return col + (h * nt + (k - a)) * ROW_BYTES;
    }
    const uint32_t nb = b > 16 ? b - max_u(a, 16) : 0;
    return col + (2 * nt + h * nb + (k - max_u(a, 16))) * ROW_BYTES;
}

// copy n rows inside L1 over the NoC: destination row k0 + i <- source row s0 + i, in runs that stay inside one
// 16-row face half on both sides (both layouts are linear in the row there); dst(k, h) / src(s, h) are row
// addresses. Caller barriers.
template <typename D, typename S>
inline void copy_rows(uint32_t k0, uint32_t s0, uint32_t n, D&& dst, S&& src) {
    while (n > 0) {
        const uint32_t len = min_u(n, min_u(16 - (k0 & 15), 16 - (s0 & 15)));
        for (uint32_t h = 0; h < 2; ++h) {
            noc_async_read(get_noc_addr(src(s0, h)), dst(k0, h), len * ROW_BYTES);
        }
        k0 += len;
        s0 += len;
        n -= len;
    }
}

struct Item {
    bool piece;  // false: D pages [j, j + B, ..), n of them; true: landing piece
    uint32_t j, n;
    uint32_t img, row, a, b;  // piece: D tile row (image-local) and my rows [a, b) in it
    uint32_t c0, c1;          // piece: columns
    uint32_t slot;            // piece: landing row index (img * nsh + rank)
};

// the wave of D tile row R (image img) for chip me: the latest wave of the y tile rows holding my rows of it
template <uint32_t wave_rows>
inline uint32_t row_wave(const Geom& g, uint32_t img, uint32_t R) {
    const uint32_t a = rows_a(g, g.me, R), b = rows_b(g, g.me, R);
    const uint32_t y0 = 32 * R + a - g.me * g.t_rows;
    const uint32_t y1 = 32 * R + b - 1 - g.me * g.t_rows;
    const uint32_t i0 = local_row(g, img * g.ht + y0 / 32), i1 = local_row(g, img * g.ht + y1 / 32);
    return max_u(i0, i1) / wave_rows;
}

// f(ci, item) for every item of one destination in wave w; `ci` counts the items of the (wave, destination).
template <uint32_t num_banks, uint32_t run_pages, uint32_t wave_rows, typename F>
inline void for_each_item(const Geom& g, uint32_t w, uint32_t& ci, F&& f) {
    const uint32_t r_lo = (g.me * g.t_rows) / 32;
    const uint32_t r_hi = ((g.me + 1) * g.t_rows - 1) / 32;
    Item it{};
    auto flush_run = [&](uint32_t img, uint32_t ra, uint32_t rb) {
        const uint32_t a = (img * g.htd + ra) * g.blk_tiles;
        const uint32_t end = (img * g.htd + rb) * g.blk_tiles;
        const uint32_t per_bank = (end - a + num_banks - 1) / num_banks;
        const uint32_t a_bank = a % num_banks;
        it.piece = false;
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
                it.j = j0 + m * num_banks;
                it.n = min_u(cnt - m, run_pages);
                f(ci++, it);
            }
        }
    };
    for (uint32_t img = 0; img < g.nimg; ++img) {
        uint32_t run_a = 0, run_n = 0;  // pending run of alone rows [run_a, run_a + run_n)
        for (uint32_t R = r_lo; R <= r_hi; ++R) {
            const bool mine = row_wave<wave_rows>(g, img, R) == w;
            const bool alone = !shared_row(g, R);
            if (mine && alone) {
                if (run_n == 0) {
                    run_a = R;
                }
                ++run_n;
                continue;
            }
            if (run_n > 0) {
                flush_run(img, run_a, run_a + run_n);
                run_n = 0;
            }
            if (mine) {  // shared: my rows of the whole tile row, run_pages columns a piece
                it.piece = true;
                it.img = img;
                it.row = R;
                it.a = rows_a(g, g.me, R);
                it.b = rows_b(g, g.me, R);
                it.slot = img * g.nsh + shared_rank(g, R);
                for (uint32_t c0 = 0; c0 < g.blk_tiles; c0 += run_pages) {
                    it.c0 = c0;
                    it.c1 = min_u(c0 + run_pages, g.blk_tiles);
                    f(ci++, it);
                }
            }
        }
        if (run_n > 0) {
            flush_run(img, run_a, run_a + run_n);
        }
    }
}

// byte offset of a piece in the destination's landing and its size
inline uint32_t piece_offset(const Geom& g, const Item& it) {
    return it.slot * g.blk_tiles * 2048 + it.a * TILE_ROW_BYTES * g.blk_tiles + it.c0 * (it.b - it.a) * TILE_ROW_BYTES;
}
inline uint32_t piece_bytes(const Item& it) { return (it.c1 - it.c0) * (it.b - it.a) * TILE_ROW_BYTES; }

}  // namespace mhc_feed
