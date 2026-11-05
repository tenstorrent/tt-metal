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

// The split, from the named compile-time args every kernel of the op gets.
template <typename Body>
inline void for_each_attn_res_block_mix_item(Body body) {
    constexpr uint32_t grid_x = get_named_compile_time_arg_val("grid_x");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t num_items = get_named_compile_time_arg_val("num_items");
    constexpr uint32_t chunks_per_row = get_named_compile_time_arg_val("chunks_per_row");
    constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");

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

// Sites in `group` when the sites are split into groups of `sites_per_group`: all but the
// last group are full. A short last group still takes a full group's slots in every circular
// buffer, so their pointers stay aligned from one item to the next.
constexpr uint32_t sites_in_group(uint32_t group, uint32_t sites_per_group, uint32_t num_sites) {
    const uint32_t first_site = group * sites_per_group;
    return first_site + sites_per_group <= num_sites ? sites_per_group : num_sites - first_site;
}

// Which data-movement core writes a group's partials: the reader the odd groups, the writer
// the even ones (see attn_res_block_mix_partials.hpp).
constexpr bool reader_writes_group(uint32_t group) { return group % 2 == 1; }
