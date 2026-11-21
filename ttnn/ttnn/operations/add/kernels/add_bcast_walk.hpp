// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// The broadcast walk of add, shared by the reader and the compute kernel so both agree on every run and every
// b window. Mirrored line for line by `bcast_runs` in add_program_artifacts.py (which sizes b's ring from it
// and counts b's reads for the tests).
//
// A cluster owns the out pages [start, start + count) (flat, tile-padded page order). The output is
// (images n, tile-rows r < Ht, tile-cols c < Wt); b's image of out image n is nb(n), which folds b's size-1
// leading dimensions to 0. A run is a stretch of consecutive out pages that read one b window; the reader
// fetches the window once when it opens, and compute holds it at b's ring front from its first run (open:
// wait) to its last (close: pop). The b tile of out page p inside its window: Row: p % Wt - lo; Col, Scalar: 0.
//
//   Scalar (over_hw, BroadcastDim::Scalar): one b tile per run of images with the same nb.
//   Col    (over_w,  BroadcastDim::Col):    one b tile per tile-row piece.
//   Row    (over_h,  BroadcastDim::Row):    b's row, held across every row of a run of images with the same nb
//                                           (head, full rows and tail share one window of Wt tiles); a run inside
//                                           one row, or a head and a tail that do not overlap, reads only its own
//                                           columns. With `uniform`, every out page is its own run and window
//                                           of one tile (the fallback when the windows' total would not fit b's
//                                           ring, which never wraps otherwise).
//
// Why windows are shaped so: a window waited at once must not straddle the ring's end. Col / Scalar windows
// are one tile (never straddle); a Row ring holds every window its cluster opens (never wraps), or the windows
// are one tile.

#pragma once

#include <cstdint>

namespace add_bcast {

// Regimes (the compile-time `bcast_mode`; the same values as MODE_* in add_program_artifacts.py).
constexpr uint32_t MODE_FLAT = 0;
constexpr uint32_t MODE_COL = 1;
constexpr uint32_t MODE_ROW = 2;
constexpr uint32_t MODE_SCALAR = 3;
constexpr uint32_t MODE_LEADING = 4;
constexpr uint32_t MODE_NUMBER = 5;

struct Geometry {
    uint32_t Ht;       // out tile-rows per image
    uint32_t Wt;       // out tile-cols
    uint32_t n_inner;  // out's innermost leading dimension
    uint32_t s_outer;  // b images per step of out's outer leading index (0: b broadcasts it)
    uint32_t s_inner;  // b images per step of out's inner leading index (0: b broadcasts it)

    uint32_t nb(uint32_t n) const { return (n / n_inner) * s_outer + (n % n_inner) * s_inner; }
};

struct Run {
    uint32_t first_page;  // first out page
    uint32_t len;         // out pages, consecutive
    bool open;            // the run's window is new: read it (reader) / wait for it (compute)
    bool close;           // last run of the window: pop it after the run (compute)
    uint32_t win_first;   // first b page of the window
    uint32_t win_tiles;   // b pages in the window
    uint32_t lo;          // Row: the window's first column (b tile of page p = p % Wt - lo)
};

inline uint32_t min_u32(uint32_t x, uint32_t y) { return x < y ? x : y; }

// One cluster's runs, in out-page order. `f(const Run&)` is called once per run.
template <uint32_t Mode, class F>
inline void for_each_run(const Geometry& g, uint32_t start, uint32_t count, bool uniform, F&& f) {
    const uint32_t Ht = g.Ht;
    const uint32_t Wt = g.Wt;
    const uint32_t img = Ht * Wt;
    const uint32_t end = start + count;
    uint32_t p = start;

    // The run of images from `from`'s on with the same nb, cut at the cluster's end.
    auto run_end = [&](uint32_t from, uint32_t& nb) {
        const uint32_t n = from / img;
        nb = g.nb(n);
        uint32_t q = (n + 1) * img;
        while (q < end && g.nb(q / img) == nb) {
            q += img;
        }
        return min_u32(q, end);
    };

    while (p < end) {
        if constexpr (Mode == MODE_SCALAR) {
            uint32_t nb;
            const uint32_t q = run_end(p, nb);
            f(Run{p, q - p, true, true, nb, 1, 0});
            p = q;
        } else if constexpr (Mode == MODE_COL) {
            const uint32_t n = p / img;
            const uint32_t r = (p / Wt) % Ht;
            const uint32_t ln = min_u32(Wt - p % Wt, end - p);
            f(Run{p, ln, true, true, g.nb(n) * Ht + r, 1, 0});
            p += ln;
        } else {  // MODE_ROW
            uint32_t nb;
            const uint32_t q = run_end(p, nb);
            const uint32_t c0 = p % Wt;
            if (uniform) {
                for (uint32_t i = p; i < q; ++i) {
                    f(Run{i, 1, true, true, nb * Wt + i % Wt, 1, i % Wt});
                }
                p = q;
                continue;
            }
            const uint32_t head = c0 ? Wt - c0 : 0;
            if (q - p <= (head ? head : Wt)) {  // one segment, inside one row: its own columns
                f(Run{p, q - p, true, true, nb * Wt + c0, q - p, c0});
                p = q;
                continue;
            }
            const uint32_t k = (q - p - head) / Wt;
            const uint32_t tail = (q - p - head) % Wt;
            if (k > 0 || tail >= c0) {
                // head, full rows, tail: one window, b's whole row.
                if (head) {
                    f(Run{p, head, true, k == 0 && tail == 0, nb * Wt, Wt, 0});
                }
                if (k) {
                    f(Run{p + head, k * Wt, head == 0, tail == 0, nb * Wt, Wt, 0});
                }
                if (tail) {
                    f(Run{p + head + k * Wt, tail, head == 0 && k == 0, true, nb * Wt, Wt, 0});
                }
            } else {
                // a head and a tail with a gap of columns between them: each reads its own.
                f(Run{p, head, true, true, nb * Wt + c0, head, c0});
                f(Run{p + head, tail, true, true, nb * Wt, tail, 0});
            }
            p = q;
        }
    }
}

}  // namespace add_bcast
