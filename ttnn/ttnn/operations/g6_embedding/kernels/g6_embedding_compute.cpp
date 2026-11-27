// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_embedding compute: tilize_block for every block of this core in ONE helper call, so the
// tilize init/uninit is paid once per core. The helper's block is one tile-row of block_col_tiles
// tiles; a block of block_tile_rows tile-rows is block_tile_rows helper blocks (the reader and
// writer always push/pop the nominal block, ragged or not).
//
// Both CBs are UInt16 with 16-bit DEST: the standard integer tilize path, which moves every bf16
// bit pattern (NaN payloads, -0.0, subnormals, inf) unchanged.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

#if defined(G6_EMBEDDING_ZONES)
#include "tools/profiler/kernel_profiler.hpp"
#define G6_ZONE(name) DeviceZoneScopedN(name)
#else
#define G6_ZONE(name)
#endif

void kernel_main() {
    constexpr uint32_t cb_gathered_rows = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_col_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t block_tile_rows = get_compile_time_arg_val(3);

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_gathered_rows, cb_output_tiles);
    {
        G6_ZONE("tilize_block");
        compute_kernel_lib::tilize<block_col_tiles, cb_gathered_rows, cb_output_tiles>(num_blocks * block_tile_rows);
    }
}
