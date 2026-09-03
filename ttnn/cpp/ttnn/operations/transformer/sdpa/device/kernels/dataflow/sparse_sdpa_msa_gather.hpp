// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Per-NoC trid ring for split block-tile gathers. Reader and writer each issue one half of a block's K/V tile
// reads, capped at K_TRID_RING outstanding transactions per NoC.
// Include after api/dataflow/dataflow_api.h and experimental_device_api.hpp (the reader/writer already do).
#pragma once

#include <stdint.h>
#include <tt-metalium/constants.hpp>

namespace sparse_sdpa_msa {

// Convert a logical sequence tile-row/feature-tile coordinate into the tiled TensorAccessor page id used by
// a paged pool flattened as [bundle][layer][kv_head]. The false branch is compile-time free for plain caches.
template <bool PagedKV, uint32_t PageSize, uint32_t NumLayers, uint32_t LayerIdx>
FORCE_INLINE uint32_t cache_tile_id(
    uint32_t contiguous_tile_id,
    uint32_t logical_tile_row,
    uint32_t feature_tile,
    uint32_t feature_tiles,
    uint32_t n_kv,
    uint32_t kv_group,
    uint32_t page_bundle_l1) {
    if constexpr (!PagedKV) {
        return contiguous_tile_id;
    } else {
        constexpr uint32_t tile_rows_per_page = PageSize / tt::constants::TILE_HEIGHT;
        const uint32_t logical_bundle = logical_tile_row / tile_rows_per_page;
        const uint32_t tile_row_in_page = logical_tile_row % tile_rows_per_page;
        const uint32_t physical_bundle = CoreLocalMem<volatile uint16_t>(page_bundle_l1)[logical_bundle];
        const uint32_t physical_page = (physical_bundle * NumLayers + LayerIdx) * n_kv + kv_group;
        return (physical_page * tile_rows_per_page + tile_row_in_page) * feature_tiles + feature_tile;
    }
}

// Per-NoC trid-ring depth. 0 = off (plain burst). Keep <= 16.
constexpr uint32_t K_TRID_RING = 8;
// Avoid compile-time division by zero when K_TRID_RING is 0.
constexpr uint32_t TRID_MOD = (K_TRID_RING == 0) ? 1u : K_TRID_RING;

// One ring per block gather. K and V reads share the ring so their traffic overlaps.
struct TridRing {
    Noc& noc;
    uint32_t issued = 0;

    template <typename Accessor>
    FORCE_INLINE void read(
        const Accessor& t, experimental::CB& cb, uint32_t tile_bytes, uint32_t page_id, uint32_t offset_bytes) {
        if constexpr (K_TRID_RING == 0) {
            noc.async_read(t, cb, tile_bytes, {.page_id = page_id}, {.offset_bytes = offset_bytes});
        } else {
            const uint32_t trid = (issued % TRID_MOD) + 1;
            if (issued >= K_TRID_RING) {
                experimental::async_read_barrier_with_trid(noc, trid);  // free this slot before reuse
            }
            experimental::set_read_trid(noc, trid);
            noc.async_read(t, cb, tile_bytes, {.page_id = page_id}, {.offset_bytes = offset_bytes});
            ++issued;
        }
    }

    FORCE_INLINE void drain() {
        if constexpr (K_TRID_RING == 0) {
            noc.async_read_barrier();
        } else {
            const uint32_t to_drain = (issued < K_TRID_RING) ? issued : K_TRID_RING;
            for (uint32_t d = 0; d < to_drain; ++d) {
                experimental::async_read_barrier_with_trid(noc, ((issued - to_drain + d) % TRID_MOD) + 1);
            }
            experimental::set_read_trid(noc, 0);  // restore untagged
            issued = 0;
        }
    }
};

}  // namespace sparse_sdpa_msa
