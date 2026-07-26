// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::operations::experimental::indexer_score::paged {

constexpr uint32_t tile_tokens = 32;
constexpr uint32_t default_bundle_tokens = 5120;
constexpr uint32_t invalid_bundle = 0xFFFFFFFFu;

struct LocalTileAddress {
    uint32_t logical_bundle;
    uint32_t bundle_tile;
};

// Physical pool shape on each SP rank:
//   [num_physical_bundles * num_layers, 1, bundle_tokens / sp, head_dim]
// The table entry selects a physical bundle; layers are bundle-major/layer-inner. Width tiles stay
// adjacent so the native tiled TensorAccessor retains its interleaved DRAM-bank mapping.
constexpr LocalTileAddress split_local_tile(uint32_t local_seq_tile, uint32_t local_bundle_tiles) {
    return {
        .logical_bundle = local_seq_tile / local_bundle_tiles,
        .bundle_tile = local_seq_tile % local_bundle_tiles,
    };
}

constexpr LocalTileAddress split_global_tile(uint32_t global_seq_tile, uint32_t global_bundle_tiles, uint32_t sp_size) {
    return {
        .logical_bundle = global_seq_tile / global_bundle_tiles,
        .bundle_tile = (global_seq_tile % global_bundle_tiles) % (global_bundle_tiles / sp_size),
    };
}

constexpr uint32_t physical_batch(uint32_t physical_bundle, uint32_t num_layers, uint32_t layer_idx) {
    return physical_bundle * num_layers + layer_idx;
}

constexpr uint32_t physical_tile_page(
    uint32_t physical_bundle,
    uint32_t num_layers,
    uint32_t layer_idx,
    uint32_t local_bundle_tiles,
    uint32_t head_dim_tiles,
    uint32_t bundle_tile,
    uint32_t dim_tile) {
    return physical_batch(physical_bundle, num_layers, layer_idx) * local_bundle_tiles * head_dim_tiles +
           bundle_tile * head_dim_tiles + dim_tile;
}

// Position-dependent SP ownership. Every logical 5120-token bundle is split into one contiguous,
// equally-sized slab per SP rank. Paging happens *inside* the owner rank.
constexpr uint32_t owner_rank(uint32_t global_seq_tile, uint32_t global_bundle_tiles, uint32_t sp_size) {
    const uint32_t tile_in_bundle = global_seq_tile % global_bundle_tiles;
    return tile_in_bundle / (global_bundle_tiles / sp_size);
}

static_assert(default_bundle_tokens / tile_tokens == 160);
static_assert(split_global_tile(39, 160, 4).logical_bundle == 0 && split_global_tile(39, 160, 4).bundle_tile == 39);
static_assert(split_global_tile(40, 160, 4).logical_bundle == 0 && split_global_tile(40, 160, 4).bundle_tile == 0);
static_assert(split_global_tile(79, 160, 4).logical_bundle == 0 && split_global_tile(79, 160, 4).bundle_tile == 39);
static_assert(split_global_tile(159, 160, 4).logical_bundle == 0 && split_global_tile(159, 160, 4).bundle_tile == 39);
static_assert(split_global_tile(160, 160, 4).logical_bundle == 1 && split_global_tile(160, 160, 4).bundle_tile == 0);
static_assert(owner_rank(/*global_seq_tile=*/79, /*global_bundle_tiles=*/160, /*sp_size=*/4) == 1);
static_assert(physical_batch(/*physical_bundle=*/3, /*num_layers=*/21, /*layer_idx=*/7) == 70);
static_assert(
    physical_tile_page(
        /*physical_bundle=*/3,
        /*num_layers=*/21,
        /*layer_idx=*/7,
        /*local_bundle_tiles=*/40,
        /*head_dim_tiles=*/4,
        /*bundle_tile=*/39,
        /*dim_tile=*/3) == 70 * 40 * 4 + 39 * 4 + 3);

}  // namespace ttnn::operations::experimental::indexer_score::paged
