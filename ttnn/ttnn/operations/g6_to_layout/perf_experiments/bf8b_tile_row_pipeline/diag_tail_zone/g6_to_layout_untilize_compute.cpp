// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Untilize compute of R3 packer_untilize and R4 sharded_in_place: untilize_block.
//
// cb_tile_window (bfp8 tile pages, reader -> compute) -> unpack to bf16 -> pack-untilize ->
// cb_row_block (bf16 tile pages, compute -> writer): per block, moved_tile_rows tile-rows of
// block_cols_tiles tiles, each landing as 32 rows of block_cols_tiles * 64 B, the row stage the
// shared row-block writer expects. The walk is the reader's and the writer's, so all three kernels
// agree on every block's page count. Pack-untilize init/uninit run once per kernel.

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
