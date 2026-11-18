// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Overlapped N-slice prelude (perf experiment nslice_prelude_overlap): the schedule compute and reader share.
//
// The reader gathers the M-block's in0 units K-block outer (unit i = K-block i / bm, tile-row i % bm; band kh = the
// kernel row of that K-block), and compute tilizes unit i into its [bm, K] slot. Before it tilizes unit i, compute has
// untilized target(i) slab batches. Milestones m[kh] are the batches band kh needs (the host's per-core maximum over
// its M-blocks of the reader's slab plan), m[3] the slab's batch count. Inside band kh the target climbs from m[kh] to
// m[kh + 1], reaching it `lead` units before the band ends, so band kh + 1 is in the slab before the reader wants it.
// Both sides evaluate the same integers, so every TRISC takes the same path.

#pragma once

#include <cstdint>

namespace conv2d_3x3::overlap {

struct Milestones {
    uint32_t m[4];
};

inline Milestones milestones(uint32_t need0, uint32_t need1, uint32_t need2, uint32_t total) {
    Milestones s;
    s.m[0] = need0 < total ? need0 : total;
    s.m[1] = need1 < s.m[0] ? s.m[0] : (need1 < total ? need1 : total);
    s.m[2] = need2 < s.m[1] ? s.m[1] : (need2 < total ? need2 : total);
    s.m[3] = total;
    return s;
}

// Untilize batches done before unit i is tilized. units_per_band > 0.
// slots > 0: band kh + 1's batches are untilized in at most `slots` chunks spread over band kh (fewer untilize <->
// tilize switches); 0: one step per unit. all_first: the whole slab before the first tilize (a single switch; the
// reader still gathers band 0's units into the ring while bands 1 and 2 are untilized).
inline uint32_t target(
    const Milestones& s, uint32_t i, uint32_t units_per_band, uint32_t lead, uint32_t slots, bool all_first) {
    if (all_first) {
        return s.m[3];
    }
    const uint32_t kh = i / units_per_band;
    const uint32_t j = i - kh * units_per_band;
    const uint32_t span = units_per_band > lead ? units_per_band - lead : 1;
    const uint32_t step = j + 1 < span ? j + 1 : span;
    const uint32_t q = slots == 0 ? span : slots;
    const uint32_t q_num = (step * q + span - 1) / span;  // in [1, q]
    const uint32_t delta = s.m[kh + 1] - s.m[kh];
    return s.m[kh] + (delta * q_num + q - 1) / q;
}

}  // namespace conv2d_3x3::overlap
