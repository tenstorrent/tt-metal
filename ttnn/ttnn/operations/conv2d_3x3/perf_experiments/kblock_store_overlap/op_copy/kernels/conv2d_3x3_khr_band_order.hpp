// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Band-first order of the kh-reuse tile-rows: the reader's unit stream (conv2d_3x3_rm_ring.hpp KhrBandRuns) and
// compute's tilize (kh_reuse::Tilize) both follow it, so the unit FIFO stays in order.
//
// Virtual tile-row j (counted from the core's first output tile-row) is in lane 0 for j < wt, lane 1 for wt <= j < 2 wt
// and lane 2 above; K-block (kh, kw) of M-block m reads rows m bm + kh wt + [0, bm). Before K-block (m, kh, 0), lanes
// kh.. are brought up to that K-block's last row (the lanes below kh hold no row it reads). M-block 0 then waits for
// its 3 bm rows only, instead of every row up to 2 wt + bm (the whole first window, which all cores fetch from DRAM at
// once), and the rows between its bands come in behind compute, bm per lane per M-block, until lanes 0 and 1 are
// complete and the order is the monotone one. Rows no K-block reads (lanes 0 and 1 past nmb bm when a core spans less
// than an image row) are never produced.

#pragma once

#include <cstdint>

namespace conv2d_3x3 {

template <uint32_t wt, uint32_t bm>
struct KhrBandOrder {
    uint32_t hw[3] = {0, wt, 2 * wt};  // lane l's next row
    uint32_t end2 = 2 * wt;            // lane 2's end: the core's last row + 1

    inline __attribute__((always_inline)) void init(uint32_t num_m_blocks) {
        hw[0] = 0;
        hw[1] = wt;
        hw[2] = 2 * wt;
        end2 = num_m_blocks * bm + 2 * wt;
    }

    // Lane l's rows needed by K-block (m, kh, 0) (m0 = m bm, lane 2 ending at e2): up to (excluding) the returned row.
    static inline __attribute__((always_inline)) uint32_t target_of(uint32_t m0, uint32_t kh, uint32_t l, uint32_t e2) {
        const uint32_t end = l == 0 ? wt : (l == 1 ? 2 * wt : e2);
        const uint32_t t = m0 + kh * wt + bm;
        return t < end ? t : end;
    }
    inline __attribute__((always_inline)) uint32_t target(uint32_t m, uint32_t kh, uint32_t l) const {
        return target_of(m * bm, kh, l, end2);
    }

    // Rows the order produces in total.
    static inline __attribute__((always_inline)) uint32_t count(uint32_t num_m_blocks) {
        const uint32_t rows = num_m_blocks * bm;
        return 2 * (rows < wt ? rows : wt) + rows;
    }
};

}  // namespace conv2d_3x3
