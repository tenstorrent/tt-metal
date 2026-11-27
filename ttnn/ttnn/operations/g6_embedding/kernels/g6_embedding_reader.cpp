// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_embedding reader (RISCV_1, NoC0): fill_index_window + gather_rows_block.
//
// Raw NoC rather than dataflow_kernel_lib::read_sticks_for_tilize: that helper reads consecutive
// pages, while every gathered row's page id is an index value loaded from L1, and the window fill
// reads a byte sub-range of one index page into a scratch CB.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_index_window = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gathered_rows = get_compile_time_arg_val(1);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t width_blocks_per_row = get_compile_time_arg_val(3);
    constexpr uint32_t block_tile_rows = get_compile_time_arg_val(4);
    constexpr uint32_t tile_rows_per_image = get_compile_time_arg_val(5);
    constexpr uint32_t tokens_per_image = get_compile_time_arg_val(6);
    constexpr uint32_t index_window_tile_rows = get_compile_time_arg_val(7);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t index_page_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t weight_page_bytes = get_compile_time_arg_val(10);
    constexpr auto index_args = TensorAccessorArgs<11>();
    constexpr auto weight_args = TensorAccessorArgs<index_args.next_compile_time_args_offset()>();

    constexpr uint32_t tile_height = 32;
    constexpr uint32_t index_tile_row_bytes = tile_height * sizeof(uint32_t);
    constexpr uint32_t index_image_bytes = tokens_per_image * sizeof(uint32_t);
    // One position's slice of a table row: block_width_tiles tile columns; a unit is 32 such
    // sticks, i.e. block_width_tiles pages of cb_gathered_rows.
    constexpr uint32_t row_slice_bytes = block_width_tiles * (tile_bytes / tile_height);
    constexpr uint32_t unit_pages = block_width_tiles;
    constexpr uint32_t unit_bytes = unit_pages * tile_bytes;

    const uint32_t index_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t weight_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t first_unit = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);

    const auto index_accessor = TensorAccessor(index_args, index_addr, index_page_bytes);
    const auto weight_accessor = TensorAccessor(weight_args, weight_addr, weight_page_bytes);

    const uint32_t end_unit = first_unit + num_units;
    const uint32_t core_end_tile_row = (end_unit - 1) / width_blocks_per_row + 1;

    // The window is scratch: reserved once, read in place, never pushed.
    cb_reserve_back(cb_index_window, index_window_tile_rows);
    const uint32_t window_base = get_write_ptr(cb_index_window);
    uint32_t window_first_tile_row = 0;
    uint32_t window_end_tile_row = 0;

    // Reads the indices of tile rows [begin, end) into the window, one read per image segment.
    // Bytes past an image's last token are never read, so they never address the table.
    auto fill_index_window = [&](uint32_t begin, uint32_t end) {
        uint32_t tile_row = begin;
        while (tile_row < end) {
            const uint32_t image = tile_row / tile_rows_per_image;
            const uint32_t h = tile_row - image * tile_rows_per_image;
            uint32_t segment_end = (image + 1) * tile_rows_per_image;
            if (segment_end > end) {
                segment_end = end;
            }
            const uint32_t byte_begin = h * index_tile_row_bytes;
            uint32_t byte_end = (h + segment_end - tile_row) * index_tile_row_bytes;
            if (byte_end > index_image_bytes) {
                byte_end = index_image_bytes;
            }
            noc_async_read(
                index_accessor.get_noc_addr(image, byte_begin),
                window_base + (tile_row - begin) * index_tile_row_bytes,
                byte_end - byte_begin);
            tile_row = segment_end;
        }
        noc_async_read_barrier();
    };

    uint32_t tile_row = first_unit / width_blocks_per_row;
    uint32_t width_block = first_unit % width_blocks_per_row;

    for (uint32_t block_first = first_unit; block_first < end_unit; block_first += block_tile_rows) {
        const uint32_t units_left = end_unit - block_first;
        const uint32_t block_units = units_left < block_tile_rows ? units_left : block_tile_rows;
        const uint32_t block_last_tile_row = (block_first + block_units - 1) / width_blocks_per_row;

        if (block_last_tile_row >= window_end_tile_row) {
            DeviceZoneScopedN("IDX_FILL");
            window_first_tile_row = tile_row;
            window_end_tile_row = window_first_tile_row + index_window_tile_rows;
            if (window_end_tile_row > core_end_tile_row) {
                window_end_tile_row = core_end_tile_row;
            }
            fill_index_window(window_first_tile_row, window_end_tile_row);
        }

        {
            DeviceZoneScopedN("GATHER_SLOT_WAIT");
            cb_reserve_back(cb_gathered_rows, block_units * unit_pages);
        }
        {
            DeviceZoneScopedN("GATHER_ISSUE");
            uint32_t unit_slot = get_write_ptr(cb_gathered_rows);
            for (uint32_t u = 0; u < block_units; ++u) {
                const uint32_t h = tile_row % tile_rows_per_image;
                const uint32_t tokens_left = tokens_per_image - h * tile_height;
                const uint32_t rows_valid = tokens_left < tile_height ? tokens_left : tile_height;
                volatile tt_l1_ptr uint32_t* indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    window_base + (tile_row - window_first_tile_row) * index_tile_row_bytes);
                const uint32_t column_offset = width_block * row_slice_bytes;

                // Sticks r >= rows_valid stay unwritten: they become the padding rows of the
                // image's partial last tile row, whose content the contract leaves unspecified.
                uint32_t stick = unit_slot;
                for (uint32_t r = 0; r < rows_valid; ++r) {
                    noc_async_read(weight_accessor.get_noc_addr(indices[r], column_offset), stick, row_slice_bytes);
                    stick += row_slice_bytes;
                }

                unit_slot += unit_bytes;
                if (++width_block == width_blocks_per_row) {
                    width_block = 0;
                    ++tile_row;
                }
            }
        }
        {
            DeviceZoneScopedN("GATHER_WAIT");
            noc_async_read_barrier();
        }
        for (uint32_t u = 0; u < block_units; ++u) {
            cb_push_back(cb_gathered_rows, unit_pages);
        }
    }
}
