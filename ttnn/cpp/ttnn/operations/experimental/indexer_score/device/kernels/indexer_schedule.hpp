// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "indexer_ring_schedule.hpp"

namespace indexer_schedule {
struct CoreSchedule {
    uint32_t row;
    uint32_t col;
    uint32_t block;
    uint32_t row_group;
    uint32_t band_start;  // contiguous band offset, or ring lane identity
    uint32_t band_count;
};

// Shared core identity layout. Unfused work retains the two-level contiguous
// block/column split; fused work retains the ring-arrival lane distribution.
template <bool Ring>
inline constexpr CoreSchedule for_core(
    uint32_t core_id, uint32_t group_rows, const indexer_ring_schedule::Geometry& geometry) {
    const uint32_t row = core_id / geometry.cols;
    const uint32_t col = core_id % geometry.cols;
    const uint32_t block = row / group_rows;
    uint32_t start = 0, count = 0;
    if constexpr (Ring) {
        start = block + col * geometry.num_blocks;
        count = indexer_ring_schedule::band_count(geometry, start);
    } else {
        const uint32_t per_block = geometry.units_per_shard / geometry.num_blocks;
        const uint32_t extra_blocks = geometry.units_per_shard % geometry.num_blocks;
        const uint32_t block_start = block * per_block + (block < extra_blocks ? block : extra_blocks);
        const uint32_t block_size = per_block + (block < extra_blocks);
        const uint32_t per_col = block_size / geometry.cols;
        const uint32_t extra_cols = block_size % geometry.cols;
        start = block_start + col * per_col + (col < extra_cols ? col : extra_cols);
        count = per_col + (col < extra_cols);
    }
    return {row, col, block, row % group_rows, start, count};
}
}  // namespace indexer_schedule
