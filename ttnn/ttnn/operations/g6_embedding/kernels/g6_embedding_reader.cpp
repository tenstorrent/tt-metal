// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_embedding reader: fetch_index_block + gather_rows_block.
//
// Raw NoC reads instead of dataflow_kernel_lib::read_sticks_for_tilize: that helper addresses
// contiguous sticks (page id = start + row), while here every row's page id is an index value
// loaded from L1 at run time. Its reserve -> reads -> one barrier -> push structure is kept.
//
// cb_index_values is reader-private scratch (no CB credit): INDEX_DEPTH slot-groups of
// BLOCK_TILE_ROWS 128-B slots, ordered by this kernel's own read barriers. The next row-block's
// indices are fetched after the current block's row reads are issued and before its barrier, so
// one barrier retires both and index latency leaves the critical path after the first block.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tile_constants.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

using tt::constants::TILE_HEIGHT;

void kernel_main() {
    constexpr uint32_t cb_index_values = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gathered_rows = get_compile_time_arg_val(1);
    constexpr uint32_t block_col_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t block_tile_rows = get_compile_time_arg_val(3);
    constexpr uint32_t col_blocks_per_row = get_compile_time_arg_val(4);
    constexpr uint32_t index_depth = get_compile_time_arg_val(5);
    constexpr uint32_t tile_rows_per_image = get_compile_time_arg_val(6);
    constexpr uint32_t tensor_token_extent = get_compile_time_arg_val(7);
    constexpr uint32_t tensor_tile_rows = get_compile_time_arg_val(8);
    constexpr uint32_t row_segment_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t index_slot_bytes = get_compile_time_arg_val(10);
    constexpr auto index_args = TensorAccessorArgs<11>();
    constexpr auto table_args = TensorAccessorArgs<index_args.next_compile_time_args_offset()>();

    static_assert(index_depth >= 2, "prefetching the next row-block needs a second slot-group");

    constexpr uint32_t block_pages = block_tile_rows * block_col_tiles;
    constexpr uint32_t tile_row_bytes = TILE_HEIGHT * row_segment_bytes;
    constexpr uint32_t slot_group_bytes = block_tile_rows * index_slot_bytes;
    constexpr uint32_t index_element_bytes = index_slot_bytes / TILE_HEIGHT;

    const uint32_t index_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t table_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t first_block = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);

    const auto index_accessor = TensorAccessor(index_args, index_addr);
    const auto table_accessor = TensorAccessor(table_args, table_addr);
    const uint32_t index_slots_base = get_write_ptr(cb_index_values);

    // Tokens of tile-row g that exist: a partial last tile-row of an image reads only these, so no
    // position past S of the index page is read and no row past it is gathered.
    auto valid_rows_of = [](uint32_t tile_row_in_image) -> uint32_t {
        const uint32_t remaining = tensor_token_extent - tile_row_in_image * TILE_HEIGHT;
        return remaining < TILE_HEIGHT ? remaining : TILE_HEIGHT;
    };

    auto fetch_index_block = [&](uint32_t row_block, uint32_t slot_group_addr) {
        const uint32_t first_tile_row = row_block * block_tile_rows;
        for (uint32_t t = 0; t < block_tile_rows; ++t) {
            const uint32_t tile_row = first_tile_row + t;
            if (tile_row >= tensor_tile_rows) {
                break;
            }
            const uint32_t image = tile_row / tile_rows_per_image;
            const uint32_t tile_row_in_image = tile_row - image * tile_rows_per_image;
            noc_async_read(
                index_accessor.get_noc_addr(image, tile_row_in_image * index_slot_bytes),
                slot_group_addr + t * index_slot_bytes,
                valid_rows_of(tile_row_in_image) * index_element_bytes);
        }
    };

    auto gather_rows_block = [&](uint32_t row_block, uint32_t col_block, uint32_t slot_group_addr, uint32_t dst) {
        const uint32_t first_tile_row = row_block * block_tile_rows;
        const uint32_t segment_offset = col_block * row_segment_bytes;
        for (uint32_t t = 0; t < block_tile_rows; ++t) {
            const uint32_t tile_row = first_tile_row + t;
            if (tile_row >= tensor_tile_rows) {
                break;
            }
            const uint32_t tile_row_in_image = tile_row % tile_rows_per_image;
            const uint32_t valid_rows = valid_rows_of(tile_row_in_image);
            volatile tt_l1_ptr uint32_t* indices =
                dataflow_kernel_lib::addr_to_l1_ptr(slot_group_addr + t * index_slot_bytes);
            uint32_t row_dst = dst + t * tile_row_bytes;
            for (uint32_t r = 0; r < valid_rows; ++r) {
                noc_async_read(table_accessor.get_noc_addr(indices[r], segment_offset), row_dst, row_segment_bytes);
                row_dst += row_segment_bytes;
            }
        }
    };

    uint32_t row_block = first_block / col_blocks_per_row;
    uint32_t col_block = first_block - row_block * col_blocks_per_row;
    uint32_t slot_group = 0;

    fetch_index_block(row_block, index_slots_base);
    noc_async_read_barrier();

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        cb_reserve_back(cb_gathered_rows, block_pages);
        gather_rows_block(
            row_block, col_block, index_slots_base + slot_group * slot_group_bytes, get_write_ptr(cb_gathered_rows));

        if (++col_block == col_blocks_per_row) {
            col_block = 0;
            ++row_block;
            if (block_idx + 1 < num_blocks) {
                slot_group = (slot_group + 1) % index_depth;
                fetch_index_block(row_block, index_slots_base + slot_group * slot_group_bytes);
            }
        }

        noc_async_read_barrier();
        // Nominal count even when the block is ragged: producer and consumer quanta stay equal and
        // the ring closes; rows past valid_rows carry stale bytes into the result's padding.
        cb_push_back(cb_gathered_rows, block_pages);
    }
}
