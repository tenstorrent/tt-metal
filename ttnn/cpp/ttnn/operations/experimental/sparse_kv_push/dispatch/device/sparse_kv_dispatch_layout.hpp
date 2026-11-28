// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// sparse_kv_dispatch host geometry: the scratch layout, the packet size and the ring span tables.
#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "ttnn/operations/experimental/sparse_kv_push/dispatch/device/kernels/kvd_common.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

// Scratch pages: own lists [port][planner] (pages_own each), one list-count page per port, then the onward queues
// [lane][direction][stream]. Ports are (lane, direction), lane-major (kvd::port_index). A lane's queues hold at most
// its rows (a row crosses a directed link at most once, and every hop of it stays on its lane) plus one END per
// generation, so each lane's queues are sized by the exact count of rows hashed to it.
struct KvDispatchScratch {
    uint32_t planners = 0;
    uint32_t lanes = 0;
    uint32_t own_cap = 0;             // entries per (port, planner) list
    uint32_t pages_own = 0;           // pages per list
    std::vector<uint32_t> pages_q;    // per lane: pages per queue
    std::vector<uint32_t> lane_base;  // per lane: its first queue page
    uint32_t count_base = 0;          // first count page
    uint32_t pages = 0;               // total

    static KvDispatchScratch make(
        uint32_t max_rows,
        uint32_t stripe_chunk,
        uint32_t num_devices,
        uint32_t planners,
        uint32_t gmax,
        uint32_t lanes) {
        KvDispatchScratch s;
        s.planners = planners;
        s.lanes = lanes;
        const uint32_t ports = kvd::kDirections * lanes;
        const uint32_t owner_blocks = (max_rows / stripe_chunk + num_devices - 1) / num_devices;
        const uint32_t owner_words = owner_blocks * (stripe_chunk / 32);  // planners are dealt 32-row words
        s.own_cap = (owner_words + planners - 1) / planners * 32;
        s.pages_own = (s.own_cap + kvd::kEntriesPerPage - 1) / kvd::kEntriesPerPage;
        s.count_base = ports * planners * s.pages_own;
        std::vector<uint32_t> rows(lanes, 0);
        for (uint32_t g = 0; g < max_rows; ++g) {
            ++rows[kvd::lane_of(g, lanes)];
        }
        uint32_t next = s.count_base + ports;
        for (uint32_t l = 0; l < lanes; ++l) {
            s.pages_q.push_back((rows[l] + gmax + 1 + kvd::kEntriesPerPage - 1) / kvd::kEntriesPerPage);
            s.lane_base.push_back(next);
            next += kvd::kDirections * kvd::kStreams * s.pages_q[l];
        }
        s.pages = next;
        return s;
    }
    uint32_t own_base(uint32_t port) const { return port * planners * pages_own; }
    uint32_t queue_page(uint32_t port, uint32_t stream) const {
        const uint32_t lane = port / kvd::kDirections, dir = port % kvd::kDirections;
        return lane_base[lane] + (dir * kvd::kStreams + stream) * pages_q[lane];
    }
};

// A ring span entry by sparse_dispatch_2d's rule (mesh_routing.cpp span_entry): the fewest edges, then the shallowest,
// then of the at most two left (sorted) the one `tie` picks. Byte: forward | backward << 4.
inline uint8_t ring_span_entry(uint32_t n, uint32_t rotated_mask, uint32_t tie) {
    std::vector<uint32_t> offs;
    for (uint32_t t = 1; t < n; ++t) {
        if (rotated_mask >> t & 1) {
            offs.push_back(t);
        }
    }
    if (offs.empty()) {
        return 0;
    }
    std::vector<std::pair<uint32_t, uint32_t>> cands;
    for (uint32_t a = 0; a < n; ++a) {
        for (uint32_t b = 0; a + b < n; ++b) {
            bool covers = true;
            for (auto o : offs) {
                covers = covers && (o <= a || n - o <= b);
            }
            if (covers) {
                cands.emplace_back(a, b);
            }
        }
    }
    uint32_t m = ~0u;
    for (auto [a, b] : cands) {
        m = std::min(m, a + b);
    }
    std::vector<std::pair<uint32_t, uint32_t>> c2;
    for (auto p : cands) {
        if (p.first + p.second == m) {
            c2.push_back(p);
        }
    }
    uint32_t mm = ~0u;
    for (auto [a, b] : c2) {
        mm = std::min(mm, std::max(a, b));
    }
    std::vector<std::pair<uint32_t, uint32_t>> c3;
    for (auto p : c2) {
        if (std::max(p.first, p.second) == mm) {
            c3.push_back(p);
        }
    }
    std::sort(c3.begin(), c3.end());
    const auto [a, b] = c3[c3.size() > 1 ? tie : 0];
    return static_cast<uint8_t>(a | (b << 4));
}

// The kernels' ring table for an axis of n chips (2^n x 2 bytes), or {0} for a line.
inline std::vector<uint8_t> ring_span_table(uint32_t n, bool wrap) {
    if (!wrap) {
        return {0};
    }
    std::vector<uint8_t> t;
    for (uint32_t rm = 0; rm < (1u << n); ++rm) {
        for (uint32_t tie = 0; tie < 2; ++tie) {
            t.push_back(ring_span_entry(n, rm, tie));
        }
    }
    return t;
}

}  // namespace ttnn::operations::experimental::sparse_kv_push
