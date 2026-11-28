// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Tilize compute of the ROW_MAJOR -> TILE regimes: tilize_block.
//
// cb_row_stage (rows as tile pages, reader -> compute) -> unpack-tilize -> pack -> cb_tile_block
// (tile pages, compute -> writer).
// packer_tilize: the stage is bf16 and the output bfp8; the packer sets each face row's shared
// exponent, which is the value conversion this stage exists for.
// row_major_to_tile: both CBs are UInt16 views of bf16 bytes, so unpack, MOVA2D and pack move 16-bit
// integers and every bf16 pattern comes out bit-identical (compute_kernel_hw_startup also keeps the
// Src zero flag disabled for a UInt16 operand).
// Every block moves the nominal block_rows_tiles tile-rows (the reader pushes them, the writer waits
// them), so the whole core's work is one helper call of that many tile-rows: tilize init/uninit run
// once per kernel.

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
