// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_to_layout compute — `untilize_block` of regime R1.
//
// ONE untilize helper call over all of this core's units: init/uninit once per kernel, the helper
// waits / pops cb_input_tiles per unit (or per DEST sub-block when block_tile_cols > 8) and pushes
// block_tile_cols row-major pages per unit to cb_output_rows. Both CBs are UInt16 (opaque 16-bit),
// so bfloat16 bits pass the unpacker / packer unchanged.

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_rows = get_compile_time_arg_val(1);
    constexpr uint32_t block_tile_cols = get_compile_time_arg_val(2);

    const uint32_t units_this_core = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_input_tiles, cb_output_rows);
    {
        DeviceZoneScopedN("untilize_block");
        compute_kernel_lib::untilize<block_tile_cols, cb_input_tiles, cb_output_rows>(units_this_core);
    }
}
