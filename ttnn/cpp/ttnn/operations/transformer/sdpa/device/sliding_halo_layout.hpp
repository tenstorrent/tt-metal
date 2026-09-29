// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include "kernels/sliding_window_work_plan.hpp"

namespace ttnn::operations::transformer::sdpa::ring_joint {

// Each Q segment needs its cyclic predecessor tail; rotated Q can need two tails. A halo wider than one Q
// slab is split across several predecessors (hops), each shipping the tail of its own slab into a
// disjoint, oldest-first block of the same compact buffer.
struct ChunkedSlidingHaloLayout {
    uint32_t q_local_tile_rows = 0;
    uint32_t halo_tile_rows = 0;
    uint32_t logical_k_tile_rows = 0;
    uint32_t ring_size = 0;
    // Circular sliding KV cache slab count; 0 = unbounded (see sliding_window_work_plan.hpp).
    uint32_t circular_kv_slab_count = 0;
    uint32_t q_start_tile = 0;

    bool uses_neighbor_halo() const;
    // What `source_device` sends to the receiver `hop` positions ahead (1 = the immediate neighbour).
    SlidingHaloSources send_sources(uint32_t source_device, uint32_t hop = 1) const;
    // Hops that need a fabric exchange. A halo spanning the whole ring includes this device's own
    // slab, which is a local cache read, so that hop is excluded.
    uint32_t remote_hop_count() const;
    // Rows shipped by, and first compact-buffer row written by, hop `hop` (1-based).
    uint32_t hop_rows(uint32_t hop) const;
    uint32_t hop_dest_row(uint32_t hop) const;
};

ChunkedSlidingHaloLayout build_chunked_sliding_halo_layout(
    uint32_t q_local_tile_rows,
    uint32_t k_chunk_tile_rows,
    uint32_t sliding_window_tokens,
    uint32_t tile_height,
    uint32_t ring_size,
    uint32_t logical_k_tile_rows,
    uint32_t circular_kv_slab_count = 0,
    std::optional<uint32_t> q_start_tile = std::nullopt);

}  // namespace ttnn::operations::transformer::sdpa::ring_joint
