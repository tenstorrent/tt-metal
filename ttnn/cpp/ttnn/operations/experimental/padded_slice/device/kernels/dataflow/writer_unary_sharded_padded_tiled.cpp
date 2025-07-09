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
    const uint32_t total_num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles_per_read = get_arg_val<uint32_t>(1);
    const uint32_t num_sticks_this_core = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_untilized_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out_id = get_compile_time_arg_val(1);
    constexpr uint32_t num_dims = get_compile_time_arg_val(2);

    const uint32_t output_coord_addr = get_arg_addr(3);
    const uint32_t output_start_in_input_addr = get_arg_addr(3 + num_dims);
    const uint32_t output_end_addr = get_arg_addr(3 + 2 * num_dims);

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
#ifdef DEBUG
    DPRINT << "total_num_tiles: " << total_num_tiles << ", num_tiles_per_read: " << num_tiles_per_read
           << ", tile_size: " << tile_size << ", read_size: " << read_size << "block row size " << block_row_size
           << ENDL();
    DPRINT << "untilized CB ID: " << cb_untilized_id << " Out CB ID: " << cb_out_id << ENDL();
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
        DPRINT << "Output Coord: " << output_coord[0] << ", " << output_coord[1] << ", " << output_coord[2] << ", "
               << output_coord[3] << ENDL();
        DPRINT << "Write Addr Offset " << write_addr - base_write_addr << ENDL();
#endif
        cb_wait_front(cb_untilized_id, num_tiles_per_read);
        uint64_t noc_read_addr = get_noc_addr(get_read_ptr(cb_untilized_id));
        noc_read_addr += read_start_offset * block_row_size;

        noc_async_read(noc_read_addr, write_addr, read_rows_size * block_row_size);
        noc_async_read_barrier();

        write_addr += read_rows_size * block_row_size;
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
