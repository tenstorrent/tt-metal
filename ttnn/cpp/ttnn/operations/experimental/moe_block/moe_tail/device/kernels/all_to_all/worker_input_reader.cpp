// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// worker_input_reader: an all-to-all worker's input reader - its column slice of the owned rows and the shared expert
// Runs on: all-to-all worker cores (plan.workers), RISCV_1 (NCRISC), NoC 1
// Responsibilities:
//  - reads my worker_columns tile columns of y_own (row-major, TILE_HEIGHT rows a row tile) and the shared expert's
//    tiles from DRAM over NoC 1, row tile by row tile, for each row block in turn (block b's rows are y_own's /
//    the shared expert's l x the block stride further)
//  - pushes them to worker_tilize_add_compute (CB_ROWS, CB_SHARED_EXPERT); skip_padding: the block's live row-tile
//    mask via CB_LIVE_ROW_TILES
//  - no semaphores: synchronises only through the CBs
// Note: NoC 1 is safe for DRAM here: y_own and the shared expert were written by an earlier program, not by this one

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_tail/device/kernels/all_to_all/padded_row_tiles.hpp"

// (moe_tail_program_factory's all_to_all_cb)
constexpr uint32_t CB_ROWS = 0, CB_SHARED_EXPERT = 2;

void kernel_main() {
    moe_block_args::load();  // (my args: this chip's block of the DRAM table, common/args_table.hpp)
    constexpr uint32_t worker_columns = get_compile_time_arg_val(0);
    constexpr bool has_shared_expert = get_compile_time_arg_val(1) != 0;
    constexpr auto rows_args = TensorAccessorArgs<2>();
    constexpr auto shared_expert_args = TensorAccessorArgs<rows_args.next_compile_time_args_offset()>();
    // row blocks: their count and, a block, y_own's / the shared expert's rows that much further in every bank (bytes)
    constexpr uint32_t row_blocks = get_compile_time_arg_val(shared_expert_args.next_compile_time_args_offset());
    constexpr uint32_t rows_block_stride_bytes =
        get_compile_time_arg_val(shared_expert_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t shared_expert_block_stride_bytes =
        get_compile_time_arg_val(shared_expert_args.next_compile_time_args_offset() + 2);
    constexpr bool skip_padding = get_named_compile_time_arg_val("skip_padding") != 0;
    // (a row-major row of one tile column: TILE_WIDTH bfloat16 values)
    constexpr uint32_t tile_row_bytes = moe_block_tiles::kBf16TileBytes / tt::constants::TILE_HEIGHT;

    const uint32_t rows_address = moe_block_args::arg(0), shared_expert_address = moe_block_args::arg(1);
    const uint32_t first_column = moe_block_args::arg(2), owned_row_tiles = moe_block_args::arg(3);
    const uint32_t hidden_tiles = moe_block_args::arg(4);
    constexpr uint32_t padding_args = 5;
    for (uint32_t block = 0; block < row_blocks; ++block) {
        const auto rows =
            TensorAccessor(rows_args, rows_address + block * rows_block_stride_bytes, hidden_tiles * tile_row_bytes);
        const auto shared_expert = TensorAccessor(
            shared_expert_args,
            shared_expert_address + block * shared_expert_block_stride_bytes,
            moe_block_tiles::kBf16TileBytes);
        uint64_t live_mask = 0;
        if constexpr (skip_padding) {  // (my live row tiles: to the compute in a CB_LIVE_ROW_TILES page, words 0 / 1)
            cb_reserve_back(CB_LIVE_ROW_TILES, 1);
            const uint32_t page = get_write_ptr(CB_LIVE_ROW_TILES);
            live_mask = read_live_row_tile_mask(padding_args, page, owned_row_tiles, block);
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page)[0] = static_cast<uint32_t>(live_mask);
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page)[1] = static_cast<uint32_t>(live_mask >> 32);
            cb_push_back(CB_LIVE_ROW_TILES, 1);
        }
        for (uint32_t row_tile = 0; row_tile < owned_row_tiles; ++row_tile) {
            if (!row_tile_is_live<skip_padding>(live_mask, row_tile)) {
                continue;
            }
            cb_reserve_back(CB_ROWS, worker_columns);
            const uint32_t rows_l1 = get_write_ptr(CB_ROWS);
            for (uint32_t i = 0; i < tt::constants::TILE_HEIGHT; ++i) {
                noc_async_read(
                    rows.get_noc_addr(row_tile * tt::constants::TILE_HEIGHT + i, first_column * tile_row_bytes),
                    rows_l1 + i * worker_columns * tile_row_bytes,
                    worker_columns * tile_row_bytes);
            }
            if constexpr (has_shared_expert) {
                cb_reserve_back(CB_SHARED_EXPERT, worker_columns);
                const uint32_t shared_expert_l1 = get_write_ptr(CB_SHARED_EXPERT);
                for (uint32_t column = 0; column < worker_columns; ++column) {
                    noc_async_read(
                        shared_expert.get_noc_addr(row_tile * hidden_tiles + first_column + column),
                        shared_expert_l1 + column * moe_block_tiles::kBf16TileBytes,
                        moe_block_tiles::kBf16TileBytes);
                }
            }
            noc_async_read_barrier();
            cb_push_back(CB_ROWS, worker_columns);
            if constexpr (has_shared_expert) {
                cb_push_back(CB_SHARED_EXPERT, worker_columns);
            }
        }
    }
}
