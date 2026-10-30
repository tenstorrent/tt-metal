// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The work split, derived on the core from its own position so that a launch ships no
// per-core runtime args. A work item is a token row-tile and a group of read sites; the
// cores take items round-robin in row-major order.
//
// Include after the kernel's API header, which supplies the logical coordinates.

#pragma once

#include <cstdint>

struct AttnResScoresItem {
    uint32_t row;         // token row-tile
    uint32_t first_site;  // the group's first site
    uint32_t num_sites;   // sites in the group, short for the last group
};

// The split's compile-time args start at `Base`, in the order the program factory
// appends them: grid width, cores, items, site groups, sites per group.
template <uint32_t Base, uint32_t NumSites, typename Body>
inline void for_each_attn_res_scores_item(Body body) {
    constexpr uint32_t grid_x = get_compile_time_arg_val(Base);
    constexpr uint32_t num_cores = get_compile_time_arg_val(Base + 1);
    constexpr uint32_t num_items = get_compile_time_arg_val(Base + 2);
    constexpr uint32_t num_groups = get_compile_time_arg_val(Base + 3);
    constexpr uint32_t sites_per_group = get_compile_time_arg_val(Base + 4);

    const uint32_t core = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    for (uint32_t item = core; item < num_items; item += num_cores) {
        const uint32_t group = item % num_groups;
        const uint32_t first_site = group * sites_per_group;
        const uint32_t remaining = NumSites > first_site ? NumSites - first_site : 0;
        body(AttnResScoresItem{
            .row = item / num_groups,
            .first_site = first_site,
            .num_sites = remaining < sites_per_group ? remaining : sites_per_group});
    }
}
