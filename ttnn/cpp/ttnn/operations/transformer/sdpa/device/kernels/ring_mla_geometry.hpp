// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::operations::transformer::sdpa::ring_joint {

// Dense full-mesh chunked MLA geometry, in tiles. Tensor ranks are row-major
// cache placement ranks. Transport ranks must be translated by the route first.
// Q ownership is separate from K striping: TP lanes may own different Q heads.
// The host additionally validates mesh-axis placement and supported op modes.
struct RingMLAGeometry {
    uint32_t q_shards;
    uint32_t kv_sources;
    uint32_t q_slab_tiles;
    uint32_t source_capacity_tiles;

    constexpr bool valid() const {
        if (q_shards == 0 || kv_sources == 0 || kv_sources > 32 || kv_sources % q_shards != 0 || q_slab_tiles == 0 ||
            source_capacity_tiles == 0) {
            return false;
        }
        const uint32_t stripes = kv_sources / q_shards;
        if (q_slab_tiles % stripes != 0) {
            return false;
        }
        const uint32_t region = q_slab_tiles / stripes;
        // Aggregate capacity alone does not prove a block-cyclic cache can
        // represent each global token: every source must hold whole regions.
        return source_capacity_tiles % region == 0 && q_slab_tiles <= UINT32_MAX / q_shards &&
               source_capacity_tiles <= UINT32_MAX / kv_sources;
    }

    // All accessors require valid(). Indices must lie in the allocated geometry.
    constexpr uint32_t stripes_per_q_rank() const { return kv_sources / q_shards; }
    constexpr uint32_t region_tiles() const { return q_slab_tiles / stripes_per_q_rank(); }
    constexpr uint32_t global_chunk_tiles() const { return q_shards * q_slab_tiles; }
    constexpr uint32_t capacity_tiles() const { return kv_sources * source_capacity_tiles; }
    constexpr uint32_t q_rank(uint32_t tensor_rank) const { return tensor_rank / stripes_per_q_rank(); }
    constexpr uint32_t global_k_tile(uint32_t tensor_rank, uint32_t local_tile) const {
        const uint32_t region = region_tiles();
        return (local_tile / region) * global_chunk_tiles() + tensor_rank * region + local_tile % region;
    }

    // Metadata carries the prior prefix, not the true valid end of a partial
    // current chunk. Saturation also avoids overflow when prefix is very large.
    constexpr uint32_t metadata_end_tiles(uint32_t prefix_tiles) const {
        const uint32_t capacity = capacity_tiles();
        if (prefix_tiles >= capacity) {
            return capacity;
        }
        const uint32_t remaining = capacity - prefix_tiles;
        return prefix_tiles + (remaining < global_chunk_tiles() ? remaining : global_chunk_tiles());
    }
};

}  // namespace ttnn::operations::transformer::sdpa::ring_joint
