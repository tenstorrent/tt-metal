// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_to_layout compute: untilize_block over every tile-row of this core, one helper call.
//
// Both CBs are UInt16: the bf16 bytes go through unpack -> MOVA2D -> pack_untilize as opaque
// 16-bit data, so the Src zero flush (which would turn subnormals and -0.0 into +0.0 for a
// float format) is disabled at hw configure.

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_tiles_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rows_out = get_compile_time_arg_val(1);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(2);

    const uint32_t num_tile_rows = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_tiles_in, cb_rows_out);
    DataflowBuffer tiles_in(cb_tiles_in);
    DataflowBuffer rows_out(cb_rows_out);

    DeviceZoneScopedN("g6_compute");
    compute_kernel_lib::untilize<block_width_tiles, cb_tiles_in, cb_rows_out>(tiles_in, rows_out, num_tile_rows);
}
