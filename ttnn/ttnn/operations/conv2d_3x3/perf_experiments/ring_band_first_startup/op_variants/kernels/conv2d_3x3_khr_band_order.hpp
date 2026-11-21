// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Band-first order of the kh-reuse tile-rows (KHR_BAND_FIRST): the reader's unit stream and compute's tilize both
// follow it, so the unit FIFO stays in order.
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

    // Lane l's rows: [lane_lo(l), lane_end(l)).
#ifdef KHR_BAND_MONO
    // Before K-block (m, kh, 0) the lanes from first_lane(kh) on are brought up to it.
    static constexpr uint32_t first_lane(uint32_t) { return 0; }
    static inline __attribute__((always_inline)) uint32_t lane_lo(uint32_t l, uint32_t e2) { return l == 0 ? 0 : e2; }
    static inline __attribute__((always_inline)) uint32_t lane_end(uint32_t, uint32_t e2) { return e2; }
#else
    static constexpr uint32_t first_lane(uint32_t kh) { return kh; }
    static inline __attribute__((always_inline)) uint32_t lane_lo(uint32_t l, uint32_t) { return l * wt; }
    static inline __attribute__((always_inline)) uint32_t lane_end(uint32_t l, uint32_t e2) {
        return l == 0 ? wt : (l == 1 ? 2 * wt : e2);
    }
#endif

    inline __attribute__((always_inline)) void init(uint32_t num_m_blocks) {
        end2 = num_m_blocks * bm + 2 * wt;
        hw[0] = lane_lo(0, end2);
        hw[1] = lane_lo(1, end2);
        hw[2] = lane_lo(2, end2);
    }

    // Lane l's rows needed by K-block (m, kh, 0) (m0 = m bm, lane 2 ending at e2): up to (excluding) the returned row.
    static inline __attribute__((always_inline)) uint32_t target_of(uint32_t m0, uint32_t kh, uint32_t l, uint32_t e2) {
        const uint32_t end = lane_end(l, e2);
        const uint32_t t = m0 + kh * wt + bm;
        return t < end ? t : end;
    }
    inline __attribute__((always_inline)) uint32_t target(uint32_t m, uint32_t kh, uint32_t l) const {
        return target_of(m * bm, kh, l, end2);
    }

    // Rows the order produces in total.
    static inline __attribute__((always_inline)) uint32_t count(uint32_t num_m_blocks) {
        const uint32_t rows = num_m_blocks * bm;
#ifdef KHR_BAND_MONO
        return rows + 2 * wt;
#else
        return 2 * (rows < wt ? rows : wt) + rows;
#endif
    }
};

}  // namespace conv2d_3x3
