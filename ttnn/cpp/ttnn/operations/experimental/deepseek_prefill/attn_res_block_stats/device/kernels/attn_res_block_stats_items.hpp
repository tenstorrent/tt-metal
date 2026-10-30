// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The work split, derived on the core from its own position so that a launch ships no
// per-core runtime args. An item is one candidate's token row-tile; the cores take items
// round-robin in row-major order.
//
// Include after the kernel's API header, which supplies the logical coordinates.

#pragma once

#include <cstdint>

// The split's compile-time args start at `Base`: grid width, cores, items.
template <uint32_t Base, typename Body>
inline void for_each_attn_res_block_stats_item(Body body) {
    constexpr uint32_t grid_x = get_compile_time_arg_val(Base);
    constexpr uint32_t num_cores = get_compile_time_arg_val(Base + 1);
    constexpr uint32_t num_items = get_compile_time_arg_val(Base + 2);

    const uint32_t core = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    for (uint32_t item = core; item < num_items; item += num_cores) {
        body(item);
    }
}
