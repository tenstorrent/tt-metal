// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Untilize compute of the untilize regimes (R1 staged_reorder, R3 packer_untilize, R4
// sharded_in_place): untilize_block.
//
// cb_tile_window (tile pages, reader -> compute) -> unpack -> pack-untilize -> cb_row_block (tile
// pages, compute -> writer): per block, moved_tile_rows tile-rows of block_cols_tiles tiles, each
// landing as 32 rows of block_cols_tiles * 64 B, the row stage the shared row-block writer expects.
// R3/R4 unpack bfp8 to bf16. R1's CBs are UInt16 views of bf16 bytes, so unpack, MOVA2D and pack move
// 16-bit integers and every bf16 pattern comes out bit-identical (compute_kernel_hw_startup also keeps
// the Src zero flag disabled for a UInt16 operand).
// The walk is the reader's and the writer's, so all three kernels agree on every block's page count.
// Pack-untilize init/uninit run once per kernel.

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "g6_to_layout_block_walk.hpp"

using namespace g6_to_layout;
namespace ucfg = compute_kernel_lib::untilize_config;

void kernel_main() {
    constexpr uint32_t cb_tile_window = get_compile_time_arg_val(Geometry::next_ct_arg + 0);
    constexpr uint32_t cb_row_block = get_compile_time_arg_val(Geometry::next_ct_arg + 1);
    constexpr bool sharded = get_compile_time_arg_val(Geometry::next_ct_arg + 2) != 0;
    constexpr uint32_t block_cols_tiles = Geometry::block_cols_tiles;

    compute_kernel_hw_startup(cb_tile_window, cb_row_block);
    compute_kernel_lib::untilize_init<block_cols_tiles, cb_tile_window, cb_row_block>();

    CoreBlocks<sharded> blocks;
    Block block;
    while (blocks.next(block)) {
        compute_kernel_lib::untilize<
            block_cols_tiles,
            cb_tile_window,
            cb_row_block,
            ucfg::InitUninitMode::Neither,
            ucfg::WaitMode::WaitBlock,
            ucfg::ReconfigureRegisterDatatypeMode::NoReconfigure>(moved_tile_rows<sharded>(block));
    }

    compute_kernel_lib::untilize_uninit<block_cols_tiles, cb_tile_window, cb_row_block>();
}
