// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sliding_halo_layout.hpp"
#include "kernels/sliding_window_work_plan.hpp"

#include <algorithm>

namespace ttnn::operations::transformer::sdpa::ring_joint {
bool ChunkedSlidingHaloLayout::uses_neighbor_halo() const { return ring_size > 1 && halo_tile_rows > 0; }

SlidingHaloSources ChunkedSlidingHaloLayout::send_sources(uint32_t source_device, uint32_t hop) const {
    const auto q = build_chunked_q_mapping(
        q_start_tile, logical_k_tile_rows, q_local_tile_rows, ring_size, (source_device + hop) % ring_size);
    return sliding_halo_sources(q, q_local_tile_rows, ring_size, halo_tile_rows, circular_kv_slab_count, hop);
}

uint32_t ChunkedSlidingHaloLayout::remote_hop_count() const {
    return chunked_sliding_halo_remote_hop_count(halo_tile_rows, q_local_tile_rows, ring_size);
}

uint32_t ChunkedSlidingHaloLayout::hop_rows(uint32_t hop) const {
    return chunked_sliding_halo_hop_rows(halo_tile_rows, q_local_tile_rows, hop);
}

uint32_t ChunkedSlidingHaloLayout::dest_row(uint32_t source_device, uint32_t hop) const {
    return chunked_sliding_halo_block_dest_row(halo_tile_rows, q_local_tile_rows, ring_size, source_device, hop);
}

bool ChunkedSlidingHaloLayout::source_keyed() const {
    return chunked_sliding_halo_source_keyed(halo_tile_rows, q_local_tile_rows, ring_size);
}

std::vector<ChunkedSlidingHaloExchange> plan_chunked_sliding_halo_exchanges(
    const ChunkedSlidingHaloLayout& layout, uint32_t source_device, bool linear_topology, bool allow_multicast) {
    const uint32_t ring_size = layout.ring_size;
    const uint32_t remote_hops = layout.remote_hop_count();
    std::vector<ChunkedSlidingHaloExchange> exchanges;
    if (!allow_multicast || !layout.source_keyed()) {
        for (uint32_t hop = 1; hop <= remote_hops; ++hop) {
            const bool send_backward = linear_topology && source_device + hop >= ring_size;
            exchanges.push_back(ChunkedSlidingHaloExchange{
                .hop = hop,
                .hop_count = 1,
                .send_backward = send_backward,
                .multicast = false,
                .distance = send_backward ? ring_size - hop : hop,
            });
        }
        return exchanges;
    }
    // Hops 1..forward_hops land ahead of the source; first_wrapped_hop..remote_hops wrap past its end.
    const uint32_t forward_hops = std::min(remote_hops, ring_size - 1 - source_device);
    if (forward_hops > 0) {
        exchanges.push_back(ChunkedSlidingHaloExchange{
            .hop = 1, .hop_count = forward_hops, .send_backward = false, .multicast = true, .distance = 1});
    }
    const uint32_t first_wrapped_hop = ring_size - source_device;
    if (first_wrapped_hop <= remote_hops) {
        exchanges.push_back(ChunkedSlidingHaloExchange{
            .hop = first_wrapped_hop,
            .hop_count = remote_hops - first_wrapped_hop + 1,
            .send_backward = true,
            .multicast = true,
            .distance = ring_size - remote_hops,
        });
    }
    return exchanges;
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
