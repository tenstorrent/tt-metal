// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::operations::transformer::sdpa::ring_joint {

// Chunked prefill needs one cyclic predecessor tail. Device 0 receives the
// final device's tail from the preceding complete Q group.
struct ChunkedSlidingHaloLayout {
    uint32_t q_local_tile_rows = 0;
    uint32_t halo_tile_rows = 0;
    uint32_t logical_k_tile_rows = 0;
    uint32_t ring_size = 0;
    // Bounded circular sliding KV cache slab count; 0 = unbounded (see sliding_window_work_plan.hpp).
    uint32_t bounded_kv_slab_count = 0;

    bool uses_neighbor_halo() const;
    uint32_t send_tail_start_tile(uint32_t source_device) const;
};

ChunkedSlidingHaloLayout build_chunked_sliding_halo_layout(
    uint32_t q_local_tile_rows,
    uint32_t k_chunk_tile_rows,
    uint32_t sliding_window_tokens,
    uint32_t tile_height,
    uint32_t ring_size,
    uint32_t logical_k_tile_rows,
    uint32_t bounded_kv_slab_count = 0);

}  // namespace ttnn::operations::transformer::sdpa::ring_joint
