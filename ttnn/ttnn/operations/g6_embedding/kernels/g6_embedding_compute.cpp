// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute of g6_embedding: tilize_block, realized as ONE tilize helper call over every tile-row of
// the core's run, so tilize init/uninit is paid once per core.
//
// Both buffers are UInt16 and DEST is 16-bit: the helper takes the standard (integer) tilize path,
// so every bf16 bit pattern -- NaN payloads, -0.0, subnormals, infinities -- passes unchanged.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

#ifdef G6_EMBEDDING_ZONES
#include "api/debug/kernel_profiler.hpp"
#define G6_ZONE(name) DeviceZoneScopedN(name)
#else
#define G6_ZONE(name)
#endif

void kernel_main() {
    constexpr uint32_t block_col_tiles = get_arg(args::block_col_tiles);
    constexpr uint32_t block_tile_rows = get_arg(args::block_tile_rows);
    const uint32_t num_blocks = get_arg(args::num_blocks);

    compute_kernel_hw_startup(dfb::gathered_rows, dfb::output_tiles);

    G6_ZONE("tilize_block");
    // A helper block is one tile-row of block_col_tiles tiles; a work block holds block_tile_rows of them.
    compute_kernel_lib::tilize<block_col_tiles, dfb::gathered_rows, dfb::output_tiles>(num_blocks * block_tile_rows);
}
