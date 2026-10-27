// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::operations::transformer::sdpa::ring_joint {

// Split-KV (dense full-mesh chunked MLA) geometry, in tiles. Q is sharded over
// q_shards; K/V over kv_sources in a block-cyclic layout: global region g lives on
// tensor rank g % kv_sources at local offset (g / kv_sources) * region, where
// region = q_slab_tiles / (kv_sources / q_shards). Tensor ranks are row-major cache
// placement ranks; kernels translate transport ranks through the resolved route.
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
};

}  // namespace ttnn::operations::transformer::sdpa::ring_joint
