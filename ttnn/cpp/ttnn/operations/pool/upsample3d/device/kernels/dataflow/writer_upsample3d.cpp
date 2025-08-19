// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_blocks_to_read = get_arg_val<uint32_t>(1);
    uint32_t start_block_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(0);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(1);
    constexpr uint32_t scale_d = get_compile_time_arg_val(2);
    constexpr uint32_t scale_h = get_compile_time_arg_val(3);
    constexpr uint32_t scale_w = get_compile_time_arg_val(4);
    constexpr uint32_t depth = get_compile_time_arg_val(5);
    constexpr uint32_t height = get_compile_time_arg_val(6);
    constexpr uint32_t width = get_compile_time_arg_val(7);
    constexpr uint32_t block_height = get_compile_time_arg_val(8);
    constexpr uint32_t num_tiles_per_block_row = get_compile_time_arg_val(9);
    constexpr auto dst_args = TensorAccessorArgs<10>();

    const auto s0 = TensorAccessor(dst_args, dst_addr, output_page_size);

    const uint32_t in_width = width / scale_w;
    const uint32_t in_height = height / scale_h;
    const uint32_t in_depth = depth / scale_d;
    uint32_t end_block_id = start_block_id + num_blocks_to_read;

    // Current stick tracks which input stick we're processing
    uint32_t current_stick = block_height * start_block_id;

    for (uint32_t b = start_block_id; b < end_block_id; b++) {
        cb_wait_front(cb_id_out0, num_tiles_per_block_row);

        uint64_t base_l1_read_addr = get_read_ptr(cb_id_out0);

        for (uint32_t in_block_row = 0; in_block_row < block_height; ++in_block_row) {
            // Calculate 3D coordinates for current input stick
            // For 5D tensor [N, D, H, W, C], stick index = n*D*H*W + d*H*W + h*W + w
            uint32_t total_input_sticks = in_width * in_height * in_depth;
            uint32_t curr_index = current_stick % total_input_sticks;
            uint32_t curr_batch = current_stick / total_input_sticks;

            // Convert linear index to 3D coordinates (d, h, w)
            uint32_t d = curr_index / (in_width * in_height);
            uint32_t remaining = curr_index % (in_width * in_height);
            uint32_t h = remaining / in_width;
            uint32_t w = remaining % in_width;

            uint64_t read_addr = base_l1_read_addr + in_block_row * output_page_size;

            // Calculate start index in output tensor where we write this data
            // Output: [N, D*scale_d, H*scale_h, W*scale_w, C]
            uint32_t output_sticks_per_batch = width * height * depth;
            uint32_t start_index = curr_batch * output_sticks_per_batch + (scale_d * d) * (width * height) +
                                   (scale_h * h) * width + (scale_w * w);

            // 3D upsampling: replicate data scale_d x scale_h x scale_w times
            for (uint32_t dd = 0; dd < scale_d; dd++) {
                for (uint32_t hh = 0; hh < scale_h; hh++) {
                    for (uint32_t ww = 0; ww < scale_w; ww++) {
                        uint64_t offset = dd * (width * height) + hh * width + ww;
                        uint64_t dst_noc_addr = get_noc_addr((start_index + offset), s0);
                        noc_async_write(read_addr, dst_noc_addr, output_page_size);
                    }
                }
            }
            current_stick++;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, num_tiles_per_block_row);
    }
}
