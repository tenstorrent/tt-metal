// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post — stats-2d-tree combine geometry, shared by reader / compute / writer so the three
// kernels derive identical roles and slot indices from the same (q, Q, row) inputs.
//
// A row group's Q column-group cores (q = 0..Q-1) form ceil(Q / FANIN) leader groups of up to
// FANIN consecutive cores. Group g's leader is its first core (q = g * FANIN), fixed for all rows
// so each sender has exactly one leader (one credit semaphore). With more than one group the row's
// root is the leader of group (r % num_groups): roots rotate across rows. With one group the
// leader is the root.
//
// Transport (all push, fixed slot per sender, deterministic):
//   partial(r)  core q   -> leader(g)  cb_gather      slot set (r % DEPTH),          slot (q - g*FANIN)
//   lead(r)     leader g -> root(r)    cb_root_gather slot set ((r / NG) % DEPTH),   slot g
// A consumer GRANTS a slot set (credit semaphore increment on every sender) only after it has
// reserved that set in its own CB (so compute has popped the set's previous occupant). It then
// COLLECTS the set by waiting its per-set arrival semaphore to reach the cumulative count, and
// pushes the slots to compute, which sums them in slot-index order.
//
// Schedule (all three kernels): after row r's mix, a leader combines row r - LAG and a root row
// r - 2*LAG; the trailing rows are drained after the last mix.

#pragma once

#include <stdint.h>

namespace mhc_post_combine {

struct Geometry {
    uint32_t q;       // my column-group index within the row group
    uint32_t Q;       // column groups in the row group
    uint32_t fanin;   // COMBINE_FANIN
    uint32_t group;   // my leader group
    uint32_t member;  // my slot in the group's gather
    uint32_t groups;  // number of leader groups (tree width; 1 = single level)
    uint32_t size;    // senders in my group
    uint32_t slots;   // gather slots per set = min(fanin, Q)
    bool leader;      // I lead my group

    static constexpr Geometry of(uint32_t q, uint32_t Q, uint32_t fanin) {
        const uint32_t group = q / fanin;
        const uint32_t g0 = group * fanin;
        const uint32_t size = (Q - g0) < fanin ? (Q - g0) : fanin;
        return Geometry{q, Q, fanin, group, q - g0, (Q + fanin - 1) / fanin, size, Q < fanin ? Q : fanin, q == g0};
    }

    constexpr bool two_level() const { return groups > 1; }
    // Root group of row r (whose leader roots it). Single level: group 0.
    constexpr uint32_t root_group(uint32_t r) const { return r % groups; }
    constexpr bool roots(uint32_t r) const { return two_level() && leader && root_group(r) == group; }
    // Ordinal of row r among the rows rooted by root_group(r).
    constexpr uint32_t root_ordinal(uint32_t r) const { return r / groups; }
    // Per-slot arrival counters (Perf 1), 16 B apart in cb_comb_flags (same L1 address on every
    // core): leader gather [COMBINE_DEPTH][slots], then root gather [COMBINE_DEPTH][groups]. A sender
    // increments its own slot's counter, so the consumer hands slots to compute one at a time.
    constexpr uint32_t gather_flag(uint32_t set, uint32_t slot) const { return (set * slots + slot) * 16; }
    constexpr uint32_t root_flag(uint32_t depth, uint32_t set, uint32_t g) const {
        return (depth * slots + set * groups + g) * 16;
    }
};

}  // namespace mhc_post_combine
