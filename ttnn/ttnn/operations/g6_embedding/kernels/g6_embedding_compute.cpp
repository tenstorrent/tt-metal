// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_embedding compute: tilize_block over the core's units, one tile row of block_width_tiles
// tiles per unit. Both CBs are UInt16, so the helper takes the standard integer tilize path
// (16-bit DEST) and every bfloat16 bit pattern passes through unchanged.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_gathered_rows = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(2);

    const uint32_t num_units = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_gathered_rows, cb_output_tiles);
    DataflowBuffer gathered_rows(cb_gathered_rows);
    DataflowBuffer output_tiles(cb_output_tiles);

    DeviceZoneScopedN("TILIZE");
    compute_kernel_lib::tilize<block_width_tiles, cb_gathered_rows, cb_output_tiles>(
        gathered_rows, output_tiles, num_units);
}
