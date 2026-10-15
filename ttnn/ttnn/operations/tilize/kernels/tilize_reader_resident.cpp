// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tilize reader, resident input (sharded regimes, no retile): cb_rm_sticks is backed on this core's
// own input shard, which already holds the block in the layout the tilize consumes — no NoC
// traffic, just publish the whole shard. Separate from tilize_reader.cpp so the binary carries
// only this path (the general reader's launch cost dominates few-tile shards; BH, measured).
// Runtime args: [0] = core_rows_tiles (the only arg the host sends this kernel).

#include <cstdint>

void kernel_main() {
    constexpr uint32_t cb_rm_sticks = get_compile_time_arg_val(0);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(7);
    const uint32_t n_pages = get_arg_val<uint32_t>(0) * block_width_tiles;
    cb_reserve_back(cb_rm_sticks, n_pages);
    cb_push_back(cb_rm_sticks, n_pages);
}
