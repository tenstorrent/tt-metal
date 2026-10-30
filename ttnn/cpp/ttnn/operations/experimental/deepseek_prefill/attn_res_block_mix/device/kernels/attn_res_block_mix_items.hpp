// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The work split, derived on the core from its own position so that a launch ships no
// per-core runtime args. An item is a run of tiles along one token row-tile; a row is cut
// into `chunks_per_row` runs and the cores take items round-robin in row-major order.
//
// Include after the kernel's API header, which supplies the logical coordinates.

#pragma once

#include <cstdint>

struct AttnResBlockMixItem {
    uint32_t row;    // token row-tile
    uint32_t chunk;  // which run of the row; run 0 also writes the row's shift and mass
    uint32_t begin;  // first tile of the run along the row
    uint32_t end;    // one past its last

    uint32_t num_tiles() const { return end - begin; }
};

// The split's compile-time args start at `Base`: grid width, cores, items, runs per row,
// tiles per row.
template <uint32_t Base, typename Body>
inline void for_each_attn_res_block_mix_item(Body body) {
    constexpr uint32_t grid_x = get_compile_time_arg_val(Base);
    constexpr uint32_t num_cores = get_compile_time_arg_val(Base + 1);
    constexpr uint32_t num_items = get_compile_time_arg_val(Base + 2);
    constexpr uint32_t chunks_per_row = get_compile_time_arg_val(Base + 3);
    constexpr uint32_t Wt = get_compile_time_arg_val(Base + 4);

    const uint32_t core = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    for (uint32_t item = core; item < num_items; item += num_cores) {
        const uint32_t chunk = item % chunks_per_row;
        body(AttnResBlockMixItem{
            .row = item / chunks_per_row,
            .chunk = chunk,
            .begin = chunk * Wt / chunks_per_row,
            .end = (chunk + 1) * Wt / chunks_per_row});
    }
}
