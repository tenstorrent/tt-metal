// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_embedding reader (RISCV_1, NoC0): fill_index_window, then gather_rows_block per block.
//
// Raw NoC rather than dataflow_kernel_lib::read_sticks_for_tilize: that helper reads consecutive
// page ids, while the gather's page id is a per-row value loaded from L1 (idx[r]), and the window
// fill reads a byte sub-range of an index page into a scratch CB that tilize never consumes.
//
// Every DRAM read keeps local and NoC addresses congruent mod 64 (Blackhole DRAM read rule):
// index offsets are multiples of 128, row slices start at multiples of block_width_tiles * 64, and
// CB slots are multiples of the same.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/tile_constants.h"

using tt::constants::TILE_HEIGHT;

void kernel_main() {
    constexpr uint32_t cb_index_window = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gathered_rows = get_compile_time_arg_val(1);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t width_blocks_per_row = get_compile_time_arg_val(3);
    constexpr uint32_t block_tile_rows = get_compile_time_arg_val(4);
    constexpr uint32_t tile_rows_per_image = get_compile_time_arg_val(5);
    constexpr uint32_t tokens_per_image = get_compile_time_arg_val(6);
    constexpr uint32_t index_window_tile_rows = get_compile_time_arg_val(7);
    constexpr uint32_t row_slice_bytes = get_compile_time_arg_val(8);
    constexpr auto index_args = TensorAccessorArgs<9>();
    constexpr auto weight_args = TensorAccessorArgs<index_args.next_compile_time_args_offset()>();

    constexpr uint32_t index_slot_bytes = TILE_HEIGHT * sizeof(uint32_t);
    constexpr uint32_t index_row_bytes = tokens_per_image * sizeof(uint32_t);
    constexpr uint32_t unit_bytes = TILE_HEIGHT * row_slice_bytes;

    const uint32_t index_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t weight_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t first_unit = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);

    const auto index_accessor = TensorAccessor(index_args, index_addr);
    const auto weight_accessor = TensorAccessor(weight_args, weight_addr);

    const uint32_t core_tile_row_end = (first_unit + num_units - 1) / width_blocks_per_row + 1;

    // Reader-private scratch: reserved once, addressed in place, never pushed.
    cb_reserve_back(cb_index_window, index_window_tile_rows);
    const uint32_t window_base = get_write_ptr(cb_index_window);
    uint32_t window_first = first_unit / width_blocks_per_row;
    uint32_t window_end = 0;

    // Loads the indices of tile rows [first, first + window) of this core: one read per image
    // segment, bytes past the image's last token never read.
    auto fill_index_window = [&](uint32_t first) {
        DeviceZoneScopedN("IDX_FILL");
        window_first = first;
        window_end = first + index_window_tile_rows;
        if (window_end > core_tile_row_end) {
            window_end = core_tile_row_end;
        }
        uint32_t slot_l1 = window_base;
        for (uint32_t tile_row = window_first; tile_row < window_end;) {
            const uint32_t image = tile_row / tile_rows_per_image;
            const uint32_t h_first = tile_row - image * tile_rows_per_image;
            uint32_t h_end = h_first + (window_end - tile_row);
            if (h_end > tile_rows_per_image) {
                h_end = tile_rows_per_image;
            }
            const uint32_t byte_begin = h_first * index_slot_bytes;
            uint32_t byte_end = h_end * index_slot_bytes;
            if (byte_end > index_row_bytes) {
                byte_end = index_row_bytes;
            }
            noc_async_read(index_accessor.get_noc_addr(image, byte_begin), slot_l1, byte_end - byte_begin);
            slot_l1 += (h_end - h_first) * index_slot_bytes;
            tile_row += h_end - h_first;
        }
        noc_async_read_barrier();
    };

    fill_index_window(window_first);

    uint32_t unit = first_unit;
    for (uint32_t units_left = num_units; units_left > 0;) {
        const uint32_t block_units = units_left < block_tile_rows ? units_left : block_tile_rows;
        const uint32_t block_pages = block_units * block_width_tiles;
        {
            DeviceZoneScopedN("GATHER_SLOT_WAIT");
            cb_reserve_back(cb_gathered_rows, block_pages);
        }
        uint32_t unit_l1 = get_write_ptr(cb_gathered_rows);
        {
            DeviceZoneScopedN("GATHER_ISSUE");
            for (uint32_t u = 0; u < block_units; ++u, ++unit, unit_l1 += unit_bytes) {
                const uint32_t tile_row = unit / width_blocks_per_row;
                const uint32_t width_block = unit - tile_row * width_blocks_per_row;
                if (tile_row >= window_end) {
                    // Gathers in flight write cb_gathered_rows, not the window, and every index
                    // they use was already loaded into a register, so the window can be refilled.
                    fill_index_window(tile_row);
                }
                const uint32_t h = tile_row % tile_rows_per_image;
                const uint32_t tokens_left = tokens_per_image - h * TILE_HEIGHT;
                const uint32_t rows_valid = tokens_left < TILE_HEIGHT ? tokens_left : TILE_HEIGHT;
                const volatile tt_l1_ptr uint32_t* indices = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(
                    window_base + (tile_row - window_first) * index_slot_bytes);
                const uint32_t column_offset = width_block * row_slice_bytes;
                uint32_t row_l1 = unit_l1;
                // Rows past rows_valid are tile padding: left as stale L1, never read from the table.
                for (uint32_t r = 0; r < rows_valid; ++r, row_l1 += row_slice_bytes) {
                    noc_async_read(weight_accessor.get_noc_addr(indices[r], column_offset), row_l1, row_slice_bytes);
                }
            }
        }
        {
            DeviceZoneScopedN("GATHER_WAIT");
            noc_async_read_barrier();
        }
        cb_push_back(cb_gathered_rows, block_pages);
        units_left -= block_units;
    }
}
