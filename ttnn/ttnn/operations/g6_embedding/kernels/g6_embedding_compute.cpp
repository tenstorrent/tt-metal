// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tilize_block over the core's whole run: one helper call, tilize init/uninit once per core.
// Both CBs are UInt16 and DEST is 16-bit, so the helper takes the standard (non-fast) tilize path
// and every bf16 bit pattern is moved as an integer.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

#if defined(G6_DEVICE_ZONES)
#include "api/debug/kernel_profiler.hpp"
#define G6_ZONE(name) DeviceZoneScopedN(name)
#else
#define G6_ZONE(name)
#endif

void kernel_main() {
    constexpr uint32_t cb_gathered_rows = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_col_tiles = get_compile_time_arg_val(2);

    const uint32_t num_tile_rows = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_gathered_rows, cb_output_tiles);
    DataflowBuffer gathered_rows(cb_gathered_rows);
    DataflowBuffer output_tiles(cb_output_tiles);
    // One zone over the whole run: the helper's per-block CB waits are inside it.
    G6_ZONE("tilize_block");
    compute_kernel_lib::tilize<block_col_tiles, cb_gathered_rows, cb_output_tiles>(
        gathered_rows, output_tiles, num_tile_rows);
}
