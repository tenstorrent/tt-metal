// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// Row -> cache tile-row mapping shared by the update_cache reader and writer, so both walk the
// same ordered list of distinct tile-rows. Reads the CTAs both kernels declare (Wt, num_heads,
// block_size, block_size_t, cache_position_modulo).

constexpr uint32_t UPDATE_CACHE_SKIP_ROW = 0xFFFFFFFF;

// `position(s)` yields the raw target position of the user's row s (-1 skips the row).
// `page_table` is the user's page-table row; unused unless IS_PAGED_CACHE.
template <typename Position>
struct UpdateCacheRows {
    Position position;
    uint32_t batch_tile_offset;
    volatile tt_l1_ptr uint32_t* page_table;

    // First cache tile id of the tile-row row s lands in, or UPDATE_CACHE_SKIP_ROW.
    uint32_t tile_id(uint32_t s) const {
        constexpr uint32_t TILE_HEIGHT = 32;
        constexpr auto Wt = get_arg(args::Wt);
        constexpr auto cache_position_modulo = get_arg(args::cache_position_modulo);

        const uint32_t raw_position = position(s);
        if (raw_position == UPDATE_CACHE_SKIP_ROW) {
            return UPDATE_CACHE_SKIP_ROW;
        }
        // cache_position_modulo is a multiple of block_size, so wrapping keeps the intra-block offset.
        // The discarded branch is still checked, so its divisor must not be a constant 0.
        constexpr uint32_t modulo = cache_position_modulo > 0 ? cache_position_modulo : 1;
        uint32_t pos = raw_position;
        if constexpr (cache_position_modulo > 0) {
            pos = raw_position % modulo;
        }
#ifdef IS_PAGED_CACHE
        constexpr auto num_heads = get_arg(args::num_heads);
        constexpr auto block_size = get_arg(args::block_size);
        constexpr auto block_size_t = get_arg(args::block_size_t);
        const uint32_t physical_block_id = page_table[pos / block_size];
        return physical_block_id * num_heads * block_size_t * Wt + ((pos % block_size) / TILE_HEIGHT) * Wt;
#else
        return batch_tile_offset + (pos / TILE_HEIGHT) * Wt;
#endif
    }

    // Tile id of row s when s is the first row landing in its tile-row (it opens a read-modify-write
    // group that patches every later row of that tile-row), else UPDATE_CACHE_SKIP_ROW.
    uint32_t group_tile_id(uint32_t s) const {
        const uint32_t id = tile_id(s);
        if (id == UPDATE_CACHE_SKIP_ROW) {
            return UPDATE_CACHE_SKIP_ROW;
        }
        for (uint32_t r = 0; r < s; ++r) {
            if (tile_id(r) == id) {
                return UPDATE_CACHE_SKIP_ROW;
            }
        }
        return id;
    }
};
