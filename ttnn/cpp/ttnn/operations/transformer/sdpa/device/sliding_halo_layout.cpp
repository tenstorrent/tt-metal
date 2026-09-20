// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sliding_halo_layout.hpp"
#include "kernels/sliding_window_work_plan.hpp"

namespace ttnn::operations::transformer::sdpa::ring_joint {
bool ChunkedSlidingHaloLayout::uses_neighbor_halo() const { return ring_size > 1 && halo_tile_rows > 0; }

SlidingHaloSources ChunkedSlidingHaloLayout::send_sources(uint32_t source_device) const {
    const auto q = build_sliding_q_mapping(
        q_start_tile, logical_k_tile_rows, q_local_tile_rows, ring_size, (source_device + 1) % ring_size);
    return sliding_halo_sources(q, q_local_tile_rows, ring_size, halo_tile_rows, circular_kv_slab_count);
}

ChunkedSlidingHaloLayout build_chunked_sliding_halo_layout(
    uint32_t q_local_tile_rows,
    uint32_t k_chunk_tile_rows,
    uint32_t sliding_window_tokens,
    uint32_t tile_height,
    uint32_t ring_size,
    uint32_t logical_k_tile_rows,
    uint32_t circular_kv_slab_count,
    std::optional<uint32_t> q_start_tile) {
    ChunkedSlidingHaloLayout layout;
    layout.q_local_tile_rows = q_local_tile_rows;
    layout.logical_k_tile_rows = logical_k_tile_rows;
    layout.ring_size = ring_size;
    layout.circular_kv_slab_count = circular_kv_slab_count;
    const uint32_t q_group_tile_rows = q_local_tile_rows * ring_size;
    // The first complete Q group needs a neighbor halo too: devices 1..R-1 read
    // their predecessor within that group. The work plan clips the only missing
    // wrap predecessor (R-1 -> 0) at token zero.
    if (q_group_tile_rows == 0 || (!q_start_tile.has_value() && logical_k_tile_rows < q_group_tile_rows)) {
        return layout;
    }

    layout.q_start_tile = q_start_tile.value_or(logical_k_tile_rows - q_group_tile_rows);
    layout.halo_tile_rows = chunked_sliding_halo_tile_rows(sliding_window_tokens, tile_height, k_chunk_tile_rows);
    return layout;
}

}  // namespace ttnn::operations::transformer::sdpa::ring_joint
