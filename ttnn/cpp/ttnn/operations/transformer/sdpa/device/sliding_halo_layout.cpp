// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sliding_halo_layout.hpp"
#include "kernels/sliding_window_work_plan.hpp"

namespace ttnn::operations::transformer::sdpa::ring_joint {
bool ChunkedSlidingHaloLayout::uses_neighbor_halo() const { return ring_size > 1 && halo_tile_rows > 0; }

uint32_t ChunkedSlidingHaloLayout::send_tail_start_tile(uint32_t source_device) const {
    return chunked_sliding_halo_source_start_tile(
        source_device, q_local_tile_rows, ring_size, logical_k_tile_rows, halo_tile_rows);
}

ChunkedSlidingHaloLayout build_chunked_sliding_halo_layout(
    uint32_t q_local_tile_rows,
    uint32_t k_chunk_tile_rows,
    uint32_t sliding_window_tokens,
    uint32_t tile_height,
    uint32_t ring_size,
    uint32_t logical_k_tile_rows) {
    ChunkedSlidingHaloLayout layout;
    layout.q_local_tile_rows = q_local_tile_rows;
    layout.logical_k_tile_rows = logical_k_tile_rows;
    layout.ring_size = ring_size;
    const uint32_t q_group_tile_rows = q_local_tile_rows * ring_size;
    if (q_group_tile_rows == 0 || logical_k_tile_rows < 2 * q_group_tile_rows) {
        return layout;
    }

    layout.halo_tile_rows = chunked_sliding_halo_tile_rows(sliding_window_tokens, tile_height, k_chunk_tile_rows);
    return layout;
}

}  // namespace ttnn::operations::transformer::sdpa::ring_joint
