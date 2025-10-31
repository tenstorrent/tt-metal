// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"
#include "hw/inc/dataflow_api.h"
#include "hw/inc/dataflow_api_addrgen.h"
#include "debug/dprint.h"
#include "ckernel_defs.h"
#include "tt-metalium/constants.hpp"
uint32_t round_down(uint32_t value, uint32_t multiple) {
    if (value % multiple != 0) {
        value -= (value % multiple);
    }
    return value;
}
void kernel_main() {
    constexpr uint32_t cb_untilized_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_padding_id = get_compile_time_arg_val(2);
    constexpr uint32_t is_non_aligned = get_compile_time_arg_val(3);
    constexpr uint32_t num_dims = get_compile_time_arg_val(4);
    constexpr uint32_t output_elem_size = get_compile_time_arg_val(5);
    constexpr uint32_t output_row_size_bytes = get_compile_time_arg_val(6);

    const uint32_t total_num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles_per_read = get_arg_val<uint32_t>(1);
    const uint32_t num_sticks_this_core = get_arg_val<uint32_t>(2);
    const uint32_t padded_channels_elems = get_arg_val<uint32_t>(3);
    const uint32_t misalignment = get_arg_val<uint32_t>(4);
    const uint32_t output_coord_addr = get_arg_addr(5);
    const uint32_t output_start_in_input_addr = get_arg_addr(5 + num_dims);
    const uint32_t output_end_addr = get_arg_addr(5 + 2 * num_dims);

    constexpr uint32_t tile_size = get_tile_size(cb_out_id);
    const uint32_t read_size = tile_size * num_tiles_per_read;

    volatile tt_l1_ptr uint32_t* output_coord = (tt_l1_ptr uint32_t*)(output_coord_addr);

    const volatile tt_l1_ptr uint32_t* const output_start_in_input = (tt_l1_ptr uint32_t*)(output_start_in_input_addr);

    const volatile tt_l1_ptr uint32_t* const output_end = (tt_l1_ptr uint32_t*)(output_end_addr);

    const uint32_t base_write_addr = get_write_ptr(cb_out_id);
    uint32_t write_addr = base_write_addr;
    uint32_t rows_remaining = num_sticks_this_core;
    uint32_t tiles_read = 0;
    uint32_t read_addr = get_read_ptr(cb_untilized_id);
    uint32_t block_row_size = read_size / tt::constants::TILE_HEIGHT;
    const uint32_t pad_addr = get_read_ptr(cb_padding_id);
    const uint32_t output_row_size_elems = output_row_size_bytes / output_elem_size;

    const uint32_t padded_channels_bytes = padded_channels_elems * output_elem_size;

    if (padded_channels_elems > 0) {
        if constexpr (output_elem_size == 4) {
            volatile tt_l1_ptr uint32_t* pad_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pad_addr);
            for (uint32_t i = 0; i < output_row_size_elems; ++i) {
                pad_ptr[i] = 0;
            }
        } else if constexpr (output_elem_size == 2) {
            volatile tt_l1_ptr uint16_t* pad_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pad_addr);
            for (uint32_t i = 0; i < output_row_size_elems; ++i) {
                pad_ptr[i] = 0;
            }
        }
    }

    uint64_t pad_noc_addr = get_noc_addr(pad_addr + output_row_size_bytes - padded_channels_bytes);

#ifdef DEBUG
    DPRINT << "total_num_tiles: " << total_num_tiles << ", num_tiles_per_read: " << num_tiles_per_read
           << ", tile_size: " << tile_size << ", read_size: " << read_size << "block row size " << block_row_size
           << ENDL();
    DPRINT << "untilized CB ID: " << cb_untilized_id << " Out CB ID: " << cb_out_id << ENDL();
    DPRINT << "pad_addr : " << pad_addr << ", pad_noc_addr : " << pad_noc_addr
           << ", padded_channels_elems: " << padded_channels_elems << ENDL();
    DPRINT << "Unaligned " << is_non_aligned << ENDL();
    DPRINT << "Output Row Size Elems " << output_row_size_elems << " Bytes: " << output_row_size_bytes
           << ", Output Elem Size: " << output_elem_size << ENDL();
#endif

    const uint32_t output_end_width_in_input = output_end[1] + output_start_in_input[1];
    uint32_t row_count = 0;
    while (tiles_read < total_num_tiles && rows_remaining > 0) {
        uint32_t width_start_in_input = output_start_in_input[1] + output_coord[1];
        uint32_t width_tile_start_in_input = round_down(width_start_in_input, ckernel::TILE_HEIGHT);
        uint32_t width_tile_end_in_input = width_tile_start_in_input + ckernel::TILE_HEIGHT;

        uint32_t read_start_offset = width_start_in_input - width_tile_start_in_input;
        uint32_t read_rows_size = ckernel::TILE_HEIGHT - read_start_offset;
        if (width_tile_end_in_input > output_end_width_in_input) {
            read_rows_size -= (width_tile_end_in_input - output_end_width_in_input);
        }
        read_rows_size = std::min(read_rows_size, rows_remaining);
        rows_remaining -= read_rows_size;

#ifdef DEBUG
        DPRINT << "Width Start in Input: " << width_start_in_input
               << ", Width Tile Start in Input: " << width_tile_start_in_input
               << ", Read Start Offset: " << read_start_offset << ", Read Rows Size: " << read_rows_size << "Remaining "
               << rows_remaining << ENDL();

        DPRINT << "Tiles Read " << tiles_read << " Output Coord: " << output_coord[0] << ", " << output_coord[1] << ", "
               << output_coord[2] << ", " << output_coord[3] << ENDL();
        DPRINT << "Write Addr " << write_addr << ", Offset " << write_addr - base_write_addr << ENDL();

#endif

        cb_wait_front(cb_untilized_id, num_tiles_per_read);
        uint64_t noc_read_addr = get_noc_addr(get_read_ptr(cb_untilized_id));

        noc_read_addr += read_start_offset * block_row_size;
        if constexpr (is_non_aligned) {
            uint64_t current_noc_read_addr = noc_read_addr;
            uint32_t current_write_addr = write_addr;
            for (uint32_t row = 0; row < read_rows_size; row++) {
                noc_async_read(current_noc_read_addr + misalignment, current_write_addr, output_row_size_bytes);
                current_noc_read_addr += block_row_size;
                current_write_addr += output_row_size_bytes;
            }
        } else {
            noc_async_read(noc_read_addr, write_addr, read_rows_size * block_row_size);
        }
        uint32_t pad_write_addr = write_addr + output_row_size_bytes - padded_channels_bytes;
        if (padded_channels_elems > 0) {
#ifdef DEBUG
            volatile tt_l1_ptr uint16_t* pad_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                pad_addr + output_row_size_bytes - padded_channels_bytes);
            DPRINT << "Pad Data = ";
            for (uint32_t i = 0; i < padded_channels_elems; ++i) {
                DPRINT << pad_ptr[i] << " ";
            }
            DPRINT << ENDL();
            DPRINT << "Pad Write Addr : " << pad_write_addr << ENDL();
#endif
            for (uint32_t row_index = 0; row_index < read_rows_size; row_index++) {
                noc_async_read(pad_noc_addr, pad_write_addr, padded_channels_elems * output_elem_size);
                pad_write_addr += output_row_size_bytes;
            }
        }
        noc_async_read_barrier();

        write_addr += read_rows_size * output_row_size_bytes;
        cb_pop_front(cb_untilized_id, num_tiles_per_read);
        tiles_read += num_tiles_per_read;

        // output_coord keeps track of the current position in the output tensor
        // Increment the output coordinate for the next read by the size of the current read.
        output_coord[1] += read_rows_size;

        // If output_coord goes beyond output_end, reset it to 0 and increment the next dimension.
        for (uint32_t index = 1; index < num_dims - 1; index++) {
            if (output_coord[index] >= output_end[index]) {
                output_coord[index] = 0;
                output_coord[index + 1] += 1;
            }
        }
    }
    noc_async_read_barrier();
}
