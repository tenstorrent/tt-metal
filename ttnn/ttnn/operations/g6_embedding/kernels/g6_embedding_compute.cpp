// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_embedding compute: tilize_block as ONE tilize helper call over every tile-row of the core, so
// tilize init/uninit is paid once per core.
//
// Both DFBs are UInt16 under a 16-bit DEST: the helper takes the standard (integer) tilize path, so
// every bf16 bit pattern (NaN payloads, -0.0, subnormals) passes unchanged. A Float16_b DFB would
// select fast tilize, which rewrites them.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/debug/kernel_profiler.hpp"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t block_col_tiles = get_arg(args::block_col_tiles);
    const uint32_t num_tile_rows = get_arg(args::num_tile_rows);

    DataflowBuffer gathered_rows(dfb::gathered_rows);
    DataflowBuffer output_tiles(dfb::output_tiles);

    compute_kernel_hw_startup(dfb::gathered_rows, dfb::output_tiles);

    DeviceZoneScopedN("tilize_block");
    compute_kernel_lib::tilize<block_col_tiles, dfb::gathered_rows, dfb::output_tiles>(
        gathered_rows, output_tiles, num_tile_rows);
}
