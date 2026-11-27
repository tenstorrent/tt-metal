// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_embedding compute: tilize_block over all blocks of this core in ONE helper call, so tilize
// init/uninit run once per core. Both CBs are UInt16 with 16-bit DEST: the helper takes the
// standard tilize path, which moves bf16 bit patterns as integers (fast tilize exists only for
// Float32 / Float16_b and is not bit-exact for NaN payloads and -0.0).

#include <stdint.h>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_gathered_rows = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_col_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t block_tile_rows = get_compile_time_arg_val(3);

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_gathered_rows, cb_output_tiles);
    {
        DeviceZoneScopedN("tilize_block");
        compute_kernel_lib::tilize<block_col_tiles, cb_gathered_rows, cb_output_tiles>(num_blocks * block_tile_rows);
    }
}
