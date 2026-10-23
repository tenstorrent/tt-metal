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
    uint32_t band_start;  // first band, or ring lane identity
    uint32_t band_count;
    uint32_t band_stride;  // unfused: band j of this core is band_start + j * band_stride
};

// Shared core identity layout: lane = block + col * num_blocks. Unfused work deals bands round-robin over the
// lanes, so a runtime kv_len shorter than the compiled K capacity still spreads its valid prefix over every
// cell; per-cell counts differ by at most one and the widest is ceil(units / lanes). Fused work keeps the
// ring-arrival lane distribution.
template <bool Ring>
inline constexpr CoreSchedule for_core(
    uint32_t core_id, uint32_t group_rows, const indexer_ring_schedule::Geometry& geometry) {
    const uint32_t row = core_id / geometry.cols;
    const uint32_t col = core_id % geometry.cols;
    const uint32_t block = row / group_rows;
    const uint32_t lane = block + col * geometry.num_blocks;
    const uint32_t lanes = geometry.num_blocks * geometry.cols;
    uint32_t count = 0;
    if constexpr (Ring) {
        count = indexer_ring_schedule::band_count(geometry, lane);
    } else {
        count = lane < geometry.units_per_shard ? 1 + (geometry.units_per_shard - 1 - lane) / lanes : 0;
    }
    return {row, col, block, row % group_rows, lane, count, lanes};
}
}  // namespace indexer_schedule
