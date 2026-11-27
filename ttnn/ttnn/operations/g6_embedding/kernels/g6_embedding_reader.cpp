// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// fetch_index_block + gather_rows_block (op_design.md -> Block schedule).
//
// Raw NoC reads instead of dataflow_kernel_lib::read_sticks_for_tilize: that helper reads contiguous
// page ids (start_page + row), while the gather's page id is a data-dependent index value loaded
// from L1. The block structure is the helper's: reserve -> row reads -> one barrier -> push.
//
// cb_index_values is reader-private scratch: a ring of index_depth slots, one block-row of index
// values each, ordered by this kernel's own read barriers (no CB credit).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tile_constants.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

namespace {
using tt::constants::TILE_HEIGHT;
using tt::constants::TILE_WIDTH;
constexpr uint32_t INDEX_ROW_BYTES = TILE_HEIGHT * sizeof(uint32_t);
// One tile-column of one bf16 row; segment offsets and widths are multiples of it, which keeps
// every DRAM read at L1 offset == DRAM offset (mod 64) as Blackhole requires.
constexpr uint32_t TILE_COLUMN_ROW_BYTES = TILE_WIDTH * sizeof(uint16_t);
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_index_values = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gathered_rows = get_compile_time_arg_val(1);
    constexpr uint32_t block_col_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t block_tile_rows = get_compile_time_arg_val(3);
    constexpr uint32_t index_depth = get_compile_time_arg_val(4);
    constexpr auto index_args = TensorAccessorArgs<5>();
    constexpr auto weight_args = TensorAccessorArgs<index_args.next_compile_time_args_offset()>();

    constexpr uint32_t segment_bytes = block_col_tiles * TILE_COLUMN_ROW_BYTES;
    constexpr uint32_t staged_tile_row_bytes = TILE_HEIGHT * segment_bytes;
    constexpr uint32_t index_slot_bytes = block_tile_rows * INDEX_ROW_BYTES;

    const uint32_t index_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t weight_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t index_page_bytes = get_common_arg_val<uint32_t>(2);
    const uint32_t weight_page_bytes = get_common_arg_val<uint32_t>(3);
    const uint32_t token_extent = get_common_arg_val<uint32_t>(4);
    const uint32_t tile_rows_per_image = get_common_arg_val<uint32_t>(5);
    const uint32_t tensor_tile_rows = get_common_arg_val<uint32_t>(6);
    const uint32_t col_blocks_per_row = get_common_arg_val<uint32_t>(7);

    const uint32_t first_block = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);

    const auto index_accessor = TensorAccessor(index_args, index_addr, index_page_bytes);
    const auto weight_accessor = TensorAccessor(weight_args, weight_addr, weight_page_bytes);
    const uint32_t index_base = get_write_ptr(cb_index_values);

    auto tile_rows_in = [&](uint32_t block_row) -> uint32_t {
        const uint32_t left = tensor_tile_rows - block_row * block_tile_rows;
        return left < block_tile_rows ? left : block_tile_rows;
    };
    // Rows of a partial last tile-row of an image are never read: their indices are not positions.
    auto valid_rows_of = [&](uint32_t tile_row_in_image) -> uint32_t {
        const uint32_t left = token_extent - tile_row_in_image * TILE_HEIGHT;
        return left < TILE_HEIGHT ? left : TILE_HEIGHT;
    };

    // Issues the index reads of one block-row into an index slot; the caller's barrier retires them.
    auto fetch_index_block = [&](uint32_t block_row, uint32_t slot_addr) {
        const uint32_t first_tile_row = block_row * block_tile_rows;
        const uint32_t tile_rows = tile_rows_in(block_row);
        for (uint32_t t = 0; t < tile_rows; ++t) {
            const uint32_t tile_row = first_tile_row + t;
            const uint32_t image = tile_row / tile_rows_per_image;
            const uint32_t tile_row_in_image = tile_row - image * tile_rows_per_image;
            noc_async_read(
                index_accessor.get_noc_addr(image, tile_row_in_image * INDEX_ROW_BYTES),
                slot_addr + t * INDEX_ROW_BYTES,
                valid_rows_of(tile_row_in_image) * sizeof(uint32_t));
        }
    };

    uint32_t block_row = first_block / col_blocks_per_row;
    uint32_t col_block = first_block - block_row * col_blocks_per_row;
    uint32_t slot = 0;

    fetch_index_block(block_row, index_base);
    noc_async_read_barrier();

    for (uint32_t block = 0; block < num_blocks; ++block) {
        const uint32_t slot_addr = index_base + slot * index_slot_bytes;
        const uint32_t tile_rows = tile_rows_in(block_row);
        const uint32_t block_pages = tile_rows * block_col_tiles;
        const uint32_t segment_offset = col_block * segment_bytes;

        // gather_rows_block
        cb_reserve_back(cb_gathered_rows, block_pages);
        const uint32_t staged_base = get_write_ptr(cb_gathered_rows);
        for (uint32_t t = 0; t < tile_rows; ++t) {
            const uint32_t tile_row = block_row * block_tile_rows + t;
            const uint32_t tile_row_in_image = tile_row % tile_rows_per_image;
            const uint32_t valid_rows = valid_rows_of(tile_row_in_image);
            volatile tt_l1_ptr uint32_t* indices = dataflow_kernel_lib::addr_to_l1_ptr(slot_addr + t * INDEX_ROW_BYTES);
            uint32_t staged_row = staged_base + t * staged_tile_row_bytes;
            for (uint32_t r = 0; r < valid_rows; ++r) {
                noc_async_read(weight_accessor.get_noc_addr(indices[r], segment_offset), staged_row, segment_bytes);
                staged_row += segment_bytes;
            }
        }

        // The next block-row's indices ride on this block's barrier. Its slot is free: the block-row
        // that last used it has issued all its reads (index values are consumed at issue time).
        const bool last_col_block_of_row = (col_block + 1 == col_blocks_per_row);
        if (last_col_block_of_row && block + 1 < num_blocks) {
            slot = (slot + 1 == index_depth) ? 0 : slot + 1;
            fetch_index_block(block_row + 1, index_base + slot * index_slot_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_gathered_rows, block_pages);

        if (last_col_block_of_row) {
            col_block = 0;
            ++block_row;
        } else {
            ++col_block;
        }
    }
}
