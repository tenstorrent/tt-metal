// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    /*
    In the case the input was tiled, a single block refers to TILE_HEIGHT rows of data after untilization, block_height
    = TILE_HEIGHT

    In the case the input was ROW_MAJOR, a single block simply refers to a single output stick, in which case
    block_height = 1
    */
    uint32_t num_blocks_to_read = get_arg_val<uint32_t>(1);
    uint32_t start_block_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(0);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(1);
    constexpr uint32_t scale_h = get_compile_time_arg_val(2);
    constexpr uint32_t scale_w = get_compile_time_arg_val(3);
    constexpr uint32_t height = get_compile_time_arg_val(4);
    constexpr uint32_t width = get_compile_time_arg_val(5);
    constexpr uint32_t block_height = get_compile_time_arg_val(6);
    constexpr uint32_t num_tiles_per_block_row = get_compile_time_arg_val(7);

    constexpr auto dst_args = TensorAccessorArgs<8>();
    const auto s0 = TensorAccessor(dst_args, dst_addr, output_page_size);

    constexpr uint32_t in_width = width / scale_w;
    constexpr uint32_t in_height = height / scale_h;
    uint32_t end_block_id = start_block_id + num_blocks_to_read;
    // reader copied the data from DRAM to CB buffer.
    // writer copy the data from CB buffer to DRAM.

    uint32_t current_stick = block_height * start_block_id;

    for (uint32_t b = start_block_id; b < end_block_id; b++) {
        cb_wait_front(cb_id_out0, num_tiles_per_block_row);

        uint64_t base_l1_read_addr = get_read_ptr(cb_id_out0);

        for (uint32_t in_block_row = 0; in_block_row < block_height; ++in_block_row) {
            uint32_t curr_index = current_stick % (in_width * in_height);
            uint32_t curr_batch = current_stick / (in_width * in_height);
            uint32_t x = curr_index / in_width;
            uint32_t y = curr_index % in_width;

            uint64_t read_addr = base_l1_read_addr + in_block_row * output_page_size;

            // calculate the start index where writer will start writing the data.
            // total --> scale_h * scale_w times data will be written to the DRAM.
            // offset calcutes the relative position of the data in the stick.
            uint32_t start_index = curr_batch * width * height + (scale_h * x) * width + scale_w * y;

            for (uint32_t j = 0; j < scale_h; j++) {
                for (uint32_t k = 0; k < scale_w; k++) {
                    uint64_t offset = j * width + k;

                    uint64_t dst_noc_addr_1 = s0.get_noc_addr(start_index + offset);
                    noc_async_write(read_addr, dst_noc_addr_1, output_page_size);
                }
            }
            current_stick++;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, num_tiles_per_block_row);
    }
}
