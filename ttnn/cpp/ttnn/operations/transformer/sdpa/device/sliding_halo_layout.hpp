// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include "kernels/sliding_window_work_plan.hpp"

namespace ttnn::operations::transformer::sdpa::ring_joint {

// Each Q segment needs its cyclic predecessor tail; rotated Q can need two tails.
struct ChunkedSlidingHaloLayout {
    uint32_t q_local_tile_rows = 0;
    uint32_t halo_tile_rows = 0;
    uint32_t logical_k_tile_rows = 0;
    uint32_t ring_size = 0;
    // Circular sliding KV cache slab count; 0 = unbounded (see sliding_window_work_plan.hpp).
    uint32_t circular_kv_slab_count = 0;
    uint32_t q_start_tile = 0;

    bool uses_neighbor_halo() const;
    SlidingHaloSources send_sources(uint32_t source_device) const;
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
