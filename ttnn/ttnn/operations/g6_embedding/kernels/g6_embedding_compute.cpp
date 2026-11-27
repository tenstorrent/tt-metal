// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_embedding compute: tilize_block over the core's whole run as ONE helper call, so tilize
// init/uninit is paid once per core. Both CBs are UInt16 and DEST is 16-bit, which keeps the
// helper on the standard (non-float) tilize path: every bf16 bit pattern passes through unchanged.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_gathered_rows = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_col_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t block_tile_rows = get_compile_time_arg_val(3);

    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_gathered_rows, cb_output_tiles);
    DataflowBuffer gathered_rows(cb_gathered_rows);
    DataflowBuffer output_tiles(cb_output_tiles);

    DeviceZoneScopedN("tilize_block");
    // The helper's block is one tile-row; a planned block of block_tile_rows tile-rows is that many
    // helper blocks, each waiting/popping block_col_tiles pages of the reader's block push.
    compute_kernel_lib::tilize<block_col_tiles, cb_gathered_rows, cb_output_tiles>(
        gathered_rows, output_tiles, num_blocks_this_core * block_tile_rows);
}
