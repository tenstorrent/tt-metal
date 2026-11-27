// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_embedding reader (NCRISC / NoC0): fetch_index_block + gather_rows_block.
//
// A block is (row block, column block): block_tile_rows tile-rows x block_col_tiles tile-columns,
// block id = row_block * col_blocks_per_row + col_block; this core owns the contiguous id run
// [first_block, first_block + num_blocks). For each block it reads one table-row segment per
// valid token straight into the row-major staging CB, at the row's offset in the tile-row it
// belongs to, so compute can tilize the slot as is.
//
// Raw NoC reads instead of dataflow_kernel_lib::read_sticks_for_tilize: that helper's page id is
// start_page + row (contiguous sticks), while a gather's page id is an index value loaded from L1.
//
// cb_index_values is reader-private scratch (no CB credit): index_depth slots of one row block's
// indices. The next row block's indices are read after the current block's row reads are issued
// and before its barrier, so one barrier retires both and only the core's first block waits on
// an index read.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

#if defined(G6_EMBEDDING_ZONES)
#include "tools/profiler/kernel_profiler.hpp"
#define G6_ZONE(name) DeviceZoneScopedN(name)
#else
#define G6_ZONE(name)
#endif

void kernel_main() {
    constexpr uint32_t cb_index_values = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gathered_rows = get_compile_time_arg_val(1);
    constexpr uint32_t block_col_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t block_tile_rows = get_compile_time_arg_val(3);
    constexpr uint32_t index_depth = get_compile_time_arg_val(4);
    constexpr uint32_t col_blocks_per_row = get_compile_time_arg_val(5);
    constexpr uint32_t tile_rows_per_image = get_compile_time_arg_val(6);
    constexpr uint32_t token_extent = get_compile_time_arg_val(7);
    constexpr uint32_t tensor_tile_rows = get_compile_time_arg_val(8);
    constexpr auto index_args = TensorAccessorArgs<9>();
    constexpr auto weight_args = TensorAccessorArgs<index_args.next_compile_time_args_offset()>();

    constexpr uint32_t tile_height = 32;
    constexpr uint32_t tile_width = 32;
    constexpr uint32_t tile_row_index_bytes = tile_height * sizeof(uint32_t);
    constexpr uint32_t index_slot_bytes = block_tile_rows * tile_row_index_bytes;
    // Whole tile-columns of bf16: a multiple of 64 B, which keeps the DRAM source and the L1
    // destination congruent mod 64 (Blackhole read alignment).
    constexpr uint32_t row_segment_bytes = block_col_tiles * tile_width * sizeof(uint16_t);
    constexpr uint32_t tile_row_bytes = tile_height * row_segment_bytes;
    constexpr uint32_t block_pages = block_tile_rows * block_col_tiles;

    const uint32_t index_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t weight_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t first_block = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);

    const auto index_accessor = TensorAccessor(index_args, index_addr);
    const auto weight_accessor = TensorAccessor(weight_args, weight_addr);

    const uint32_t index_slots_base = get_write_ptr(cb_index_values);

    auto tile_rows_in = [](uint32_t row_block) -> uint32_t {
        const uint32_t remaining = tensor_tile_rows - row_block * block_tile_rows;
        return remaining < block_tile_rows ? remaining : block_tile_rows;
    };
    // Tokens of tile-row tr of an image; only the last tile-row of an image can be partial.
    auto valid_rows_of = [](uint32_t tr) -> uint32_t {
        const uint32_t remaining = token_extent - tr * tile_height;
        return remaining < tile_height ? remaining : tile_height;
    };

    // Reads exactly the logical indices of the row block: never past token S-1 of an image row.
    auto fetch_index_block = [&](uint32_t row_block, uint32_t slot_addr) {
        G6_ZONE("fetch_index_block");
        const uint32_t tile_rows = tile_rows_in(row_block);
        for (uint32_t t = 0; t < tile_rows; ++t) {
            const uint32_t g = row_block * block_tile_rows + t;
            const uint32_t image = g / tile_rows_per_image;
            const uint32_t tr = g - image * tile_rows_per_image;
            noc_async_read(
                index_accessor.get_noc_addr(image, tr * tile_row_index_bytes),
                slot_addr + t * tile_row_index_bytes,
                valid_rows_of(tr) * sizeof(uint32_t));
        }
    };

    // Rows past valid_rows keep stale bytes: they land in the output's tile padding only.
    auto issue_gather_rows_block = [&](uint32_t row_block, uint32_t col_block, uint32_t slot_addr) {
        G6_ZONE("gather_rows_block");
        cb_reserve_back(cb_gathered_rows, block_pages);
        const uint32_t block_l1 = get_write_ptr(cb_gathered_rows);
        const uint32_t col_offset = col_block * row_segment_bytes;
        const uint32_t tile_rows = tile_rows_in(row_block);
        for (uint32_t t = 0; t < tile_rows; ++t) {
            const uint32_t g = row_block * block_tile_rows + t;
            const uint32_t tr = g % tile_rows_per_image;
            const uint32_t valid_rows = valid_rows_of(tr);
            volatile tt_l1_ptr uint32_t* indices =
                dataflow_kernel_lib::addr_to_l1_ptr(slot_addr + t * tile_row_index_bytes);
            uint32_t dst = block_l1 + t * tile_row_bytes;
            for (uint32_t r = 0; r < valid_rows; ++r) {
                noc_async_read(weight_accessor.get_noc_addr(indices[r], col_offset), dst, row_segment_bytes);
                dst += row_segment_bytes;
            }
        }
    };

    uint32_t row_block = first_block / col_blocks_per_row;
    uint32_t col_block = first_block - row_block * col_blocks_per_row;
    uint32_t slot = 0;

    fetch_index_block(row_block, index_slots_base);
    noc_async_read_barrier();

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        issue_gather_rows_block(row_block, col_block, index_slots_base + slot * index_slot_bytes);

        if (++col_block == col_blocks_per_row) {
            col_block = 0;
            ++row_block;
            slot = (slot + 1 == index_depth) ? 0 : slot + 1;
            if (block_idx + 1 < num_blocks) {
                fetch_index_block(row_block, index_slots_base + slot * index_slot_bytes);
            }
        }

        {
            G6_ZONE("gather_barrier");
            noc_async_read_barrier();
        }
        cb_push_back(cb_gathered_rows, block_pages);
    }
}
