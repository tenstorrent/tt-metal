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

// Unfused schedule only: the k-band count actually dealt to the grid for this dispatch. The host compiles
// the grid (cols x blocks) and `units_per_shard` from the ALLOCATED K length, but only the bands holding the
// runtime valid prefix [0, kv_len_tiles) have work. Dealing all capacity bands in contiguous per-column
// ranges parks the whole valid prefix on the first column(s) when K is a capacity-sized buffer (1M-token
// cache: 1020 bands, 55 valid -> column 0 does all of them). Deal only the valid bands instead, but never
// fewer than one band per (block, column) cell: every core keeps >= 1 band, so the q/w row rendezvous and
// the CB protocol stay exactly as under the capacity schedule (cells past kv_len score nothing, as before).
// Capped at the compiled count. Equal to `units_per_shard` whenever kv_len covers the allocated K, so the
// fitted-capacity schedule is unchanged. Reader, compute and writer must all call this with the same
// runtime kv_len (the common KvLength arg) so their per-core band lists stay in lockstep.
inline constexpr uint32_t dealt_units(
    uint32_t units_per_shard, uint32_t num_blocks, uint32_t cols, uint32_t kv_len_tiles, uint32_t tiles_per_unit) {
    const uint32_t valid_units = (kv_len_tiles + tiles_per_unit - 1) / tiles_per_unit;
    const uint32_t cells = num_blocks * cols;
    const uint32_t units = valid_units > cells ? valid_units : cells;
    return units < units_per_shard ? units : units_per_shard;
}

// Widest cell's band count under the unfused split of `units` (the host's max_bands formula), used as the
// head-streaming q-mcast pad target. Runtime twin of the factory's compile-time schedule_max_bands.
inline constexpr uint32_t widest_cell_bands(uint32_t units, uint32_t num_blocks, uint32_t cols) {
    const uint32_t widest_block = (units + num_blocks - 1) / num_blocks;
    return (widest_block + cols - 1) / cols;
}

// The unfused per-core schedule for this dispatch: the compiled grid, with the band count bounded by the
// runtime valid prefix (see dealt_units). The ring schedule keeps its compiled lane geometry.
template <bool Ring>
inline constexpr CoreSchedule for_core_bounded(
    uint32_t core_id,
    uint32_t group_rows,
    indexer_ring_schedule::Geometry geometry,
    uint32_t kv_len_tiles,
    uint32_t tiles_per_unit) {
    if constexpr (!Ring) {
        geometry.units_per_shard =
            dealt_units(geometry.units_per_shard, geometry.num_blocks, geometry.cols, kv_len_tiles, tiles_per_unit);
    }
    return for_core<Ring>(core_id, group_rows, geometry);
}
}  // namespace indexer_schedule
