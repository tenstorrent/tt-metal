// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::operations::transformer::sdpa::ring_joint {

// Chunked prefill needs cyclic predecessor tails. A halo that fits in one Q slab is a single hop;
// a wider one is split across several predecessors, each shipping the tail of its own slab into
// a disjoint, oldest-first block of the same compact buffer. Device 0 receives the tails that wrap
// off the end of the ring from the preceding complete Q group.
struct ChunkedSlidingHaloLayout {
    uint32_t q_local_tile_rows = 0;
    uint32_t halo_tile_rows = 0;
    uint32_t logical_k_tile_rows = 0;
    uint32_t ring_size = 0;
    // Circular sliding KV cache slab count; 0 = unbounded (see sliding_window_work_plan.hpp).
    uint32_t circular_kv_slab_count = 0;

    bool uses_neighbor_halo() const;
    // Hops that need a fabric exchange. A halo spanning the whole ring includes this device's own
    // slab, which is a local cache read, so that hop is excluded.
    uint32_t remote_hop_count() const;
    // Rows shipped by, and first compact-buffer row written by, hop `hop` (1-based).
    uint32_t hop_rows(uint32_t hop) const;
    uint32_t hop_dest_row(uint32_t hop) const;
    uint32_t send_tail_start_tile(uint32_t source_device, uint32_t hop) const;
};

ChunkedSlidingHaloLayout build_chunked_sliding_halo_layout(
    uint32_t q_local_tile_rows,
    uint32_t k_chunk_tile_rows,
    uint32_t sliding_window_tokens,
    uint32_t tile_height,
    uint32_t ring_size,
    uint32_t logical_k_tile_rows,
    uint32_t circular_kv_slab_count = 0);

}  // namespace ttnn::operations::transformer::sdpa::ring_joint
