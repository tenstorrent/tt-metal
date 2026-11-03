// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace indexer_schedule {
struct CoreSchedule {
    uint32_t row;
    uint32_t col;
    uint32_t block;
    uint32_t row_group;
    uint32_t band_start;  // first band of this core's contiguous band range
    uint32_t band_count;
};

// Shared core identity layout: a two-level contiguous block/column split of the k-bands.
inline constexpr CoreSchedule for_core(
    uint32_t core_id, uint32_t group_rows, uint32_t num_units, uint32_t num_blocks, uint32_t cols) {
    const uint32_t row = core_id / cols;
    const uint32_t col = core_id % cols;
    const uint32_t block = row / group_rows;
    const uint32_t per_block = num_units / num_blocks;
    const uint32_t extra_blocks = num_units % num_blocks;
    const uint32_t block_start = block * per_block + (block < extra_blocks ? block : extra_blocks);
    const uint32_t block_size = per_block + (block < extra_blocks);
    const uint32_t per_col = block_size / cols;
    const uint32_t extra_cols = block_size % cols;
    const uint32_t start = block_start + col * per_col + (col < extra_cols ? col : extra_cols);
    const uint32_t count = per_col + (col < extra_cols);
    return {row, col, block, row % group_rows, start, count};
}
}  // namespace indexer_schedule
