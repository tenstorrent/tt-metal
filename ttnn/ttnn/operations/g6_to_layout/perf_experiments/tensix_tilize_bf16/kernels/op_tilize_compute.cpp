// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Tilize compute of packer_tilize (ROW_MAJOR bf16 -> TILE bfloat8_b): tilize_block.
//
// cb_row_stage (bf16 rows as tile pages, reader -> compute) -> unpack-tilize -> pack to bfp8 ->
// cb_tile_block (bfp8 tile pages, compute -> writer). The packer sets each face row's shared
// exponent, which is the value conversion this stage exists for. Every block moves the nominal
// block_rows_tiles tile-rows (the reader pushes them, the writer waits them), so the whole core's
// work is one helper call of that many tile-rows: tilize init/uninit run once per kernel.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "g6_to_layout_block_walk.hpp"

using namespace g6_to_layout;

void kernel_main() {
    constexpr uint32_t cb_row_stage = get_compile_time_arg_val(Geometry::next_ct_arg + 0);
    constexpr uint32_t cb_tile_block = get_compile_time_arg_val(Geometry::next_ct_arg + 1);

    uint32_t tile_rows = 0;
    CoreBlocks<false> blocks;
    Block block;
    while (blocks.next(block)) {
        tile_rows += Geometry::block_rows_tiles;
    }
    if (tile_rows == 0) {
        return;
    }

    compute_kernel_hw_startup(cb_row_stage, cb_tile_block);
    compute_kernel_lib::tilize<Geometry::block_cols_tiles, cb_row_stage, cb_tile_block>(tile_rows);
}
