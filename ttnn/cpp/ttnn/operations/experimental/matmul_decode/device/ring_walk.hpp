// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt-metalium/core_coord.hpp"

#include <cstdint>
#include <map>
#include <set>
#include <vector>

namespace tt::tt_metal {
class IDevice;
}

namespace ttnn::operations::experimental::matmul_decode::ring_walk {

// Closed-ring walk over S ∪ C for in0 gather. Shared by the FullWidthSharded and
// PartialWidthSharded ring gather factories. Every source injects its shard once (at t=0
// into successor); the shard rotates around the ring and stops at its origin's predecessor
// (the terminator receives but does not forward, saving one hop per launch).
//
// Per walk position p:
//   is_source[p]      -- p owns a shard
//   is_compute[p]     -- p is a compute core (needs the assembled cb_full_in0 / cb_in2)
//   own_sender_id[p]  -- if is_source, this shard's global index (position in S_cores)
//   num_recv[p]       = S - is_source[p]         (every non-own source's shard visits p)
//   num_sends[p]      = is_source[p] + num_recv[p] - is_terminator[p], where
//                       is_terminator[p] = is_source[(p+1) % W]
//   next_phys[p]      -- physical worker coord of walk[(p+1) % W]
//   arriving_ids[p]   -- source IDs in the order p receives them: walk the ring backward
//                        from p, skipping p itself
struct RingWalk {
    std::vector<CoreCoord> cores;
    std::vector<uint32_t> role;
    std::vector<uint32_t> num_recv;
    std::vector<uint32_t> num_sends;
    std::vector<CoreCoord> next_phys;
    std::vector<uint8_t> is_source;
    std::vector<uint8_t> is_compute;
    std::vector<uint32_t> own_sender_id;
    uint32_t num_shards = 0;
};

// Kernel-visible role encoding. Kept in the header so kernel .cpp files and factories agree.
enum : uint32_t {
    RG_ROLE_IDLE = 0,
    RG_ROLE_SOURCE_ONLY = 1,         // has own shard, not a compute core here
    RG_ROLE_HOP = 2,                 // no own shard (relay / compute-only)
    RG_ROLE_SOURCE_AND_COMPUTE = 3,  // has own shard AND is a compute core (overlap)
};

inline RingWalk build_ring_walk(
    tt::tt_metal::IDevice* device, const std::vector<CoreCoord>& sources, const std::vector<CoreCoord>& computes) {
    RingWalk w;
    w.num_shards = static_cast<uint32_t>(sources.size());

    // Layout: sources first (in row-major S_cores order, which is also the sender-id order),
    // then computes not already in sources. Wraparound closes the ring.
    std::map<CoreCoord, uint32_t> src_id_of;
    for (uint32_t i = 0; i < sources.size(); ++i) {
        src_id_of[sources[i]] = i;
    }

    w.cores.reserve(sources.size() + computes.size());
    for (const auto& c : sources) {
        w.cores.push_back(c);
    }
    for (const auto& c : computes) {
        if (src_id_of.find(c) == src_id_of.end()) {
            w.cores.push_back(c);
        }
    }

    std::set<CoreCoord> compute_set(computes.begin(), computes.end());
    const uint32_t W = static_cast<uint32_t>(w.cores.size());
    w.role.resize(W);
    w.num_recv.resize(W);
    w.num_sends.resize(W);
    w.next_phys.resize(W);
    w.is_source.resize(W);
    w.is_compute.resize(W);
    w.own_sender_id.resize(W, 0u);

    for (uint32_t p = 0; p < W; ++p) {
        const auto& c = w.cores[p];
        const auto it = src_id_of.find(c);
        w.is_source[p] = it != src_id_of.end() ? 1u : 0u;
        w.is_compute[p] = compute_set.count(c) ? 1u : 0u;
        if (w.is_source[p]) {
            w.own_sender_id[p] = it->second;
        }
        w.next_phys[p] = device->worker_core_from_logical_core(w.cores[(p + 1) % W]);
    }

    const uint32_t S = w.num_shards;
    for (uint32_t p = 0; p < W; ++p) {
        w.num_recv[p] = S - w.is_source[p];
        const uint32_t is_terminator = w.is_source[(p + 1) % W];
        w.num_sends[p] = w.is_source[p] + w.num_recv[p] - is_terminator;
        if (w.is_source[p] && w.is_compute[p]) {
            w.role[p] = RG_ROLE_SOURCE_AND_COMPUTE;
        } else if (w.is_source[p]) {
            w.role[p] = RG_ROLE_SOURCE_ONLY;
        } else {
            w.role[p] = RG_ROLE_HOP;
        }
    }
    return w;
}

// Sender IDs in arrival order at position p: walk the closed ring backward from p, collecting
// sources (excluding p itself). The first arrival is from the nearest upstream source, and
// so on -- the pipeline delivers each shard one hop later along the ring.
inline std::vector<uint32_t> arriving_sender_ids_at(uint32_t p, const RingWalk& w) {
    std::vector<uint32_t> out;
    const uint32_t W = static_cast<uint32_t>(w.cores.size());
    out.reserve(w.num_recv[p]);
    for (uint32_t d = 1; d < W; ++d) {
        const uint32_t q = (p + W - d) % W;
        if (w.is_source[q]) {
            out.push_back(w.own_sender_id[q]);
        }
    }
    return out;
}

}  // namespace ttnn::operations::experimental::matmul_decode::ring_walk
