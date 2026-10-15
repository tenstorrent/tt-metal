// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tilize writer, resident output (sharded regimes): cb_tiles_out is backed on this core's own
// output shard, so the compute packs straight into place; the writer only retires the pages.
// Separate from tilize_writer.cpp so the binary carries only this path (BH, measured).
// Runtime args: [0] = core_rows_tiles (the only arg the host sends this kernel).

#include <cstdint>

void kernel_main() {
    constexpr uint32_t cb_tiles_out = get_compile_time_arg_val(0);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(4);
    const uint32_t n_pages = get_arg_val<uint32_t>(0) * block_width_tiles;
    cb_wait_front(cb_tiles_out, n_pages);
    cb_pop_front(cb_tiles_out, n_pages);
}
