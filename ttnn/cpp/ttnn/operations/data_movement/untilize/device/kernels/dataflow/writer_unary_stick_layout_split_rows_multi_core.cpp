// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // run-time args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_input_blocks_to_process = get_arg_val<uint32_t>(1);
    const uint32_t height_wise_input_block_start_index = get_arg_val<uint32_t>(2);
    const uint32_t num_unpadded_cols_per_input_block = get_arg_val<uint32_t>(3);
    const uint32_t width_wise_output_block_start_index = get_arg_val<uint32_t>(4);
    const uint32_t num_cols_already_processed_in_first_output_block = get_arg_val<uint32_t>(5);

    // compile-time args
    constexpr uint32_t dfb_id_out0 = get_compile_time_arg_val(0);
    constexpr uint32_t tile_height = get_compile_time_arg_val(2);
    constexpr uint32_t num_tiles_per_input_block = get_compile_time_arg_val(3);
    constexpr uint32_t num_output_blocks_across_width = get_compile_time_arg_val(4);
    constexpr uint32_t output_element_size = get_compile_time_arg_val(5);
    constexpr uint32_t num_cols_per_input_block = get_compile_time_arg_val(6);
    constexpr uint32_t num_cols_per_output_block = get_compile_time_arg_val(7);

    constexpr auto dst_args = TensorAccessorArgs<8>();
    const auto s = TensorAccessor(dst_args, dst_addr);

    Noc noc;
    DataflowBuffer dfb_out(dfb_id_out0);

    auto write_tiles_in_current_block = [&](uint32_t block_height_index) {
        dfb_out.wait_front(num_tiles_per_input_block);

        // Base address of the row of elements we are going to be writing.
        uint32_t base_l1_read_addr = dfb_out.get_read_ptr();

        // Process each row of elements in the input block
        for (uint32_t j = 0; j < tile_height; ++j) {
            uint32_t current_l1_read_addr = base_l1_read_addr + j * num_cols_per_input_block * output_element_size;

            // Output page_id that we are going to be writing to
            uint32_t num_rows_already_processed = block_height_index * tile_height + j;
            uint32_t num_pages_already_processed_in_previous_rows =
                num_rows_already_processed * num_output_blocks_across_width;
            uint32_t output_page_id =
                num_pages_already_processed_in_previous_rows + width_wise_output_block_start_index;

            uint32_t num_cols_remaining_in_current_output_block =
                num_cols_per_output_block - num_cols_already_processed_in_first_output_block;
            uint32_t output_offset_within_page_in_bytes =
                num_cols_already_processed_in_first_output_block * output_element_size;

            // Iterate through all columns in the input block that this core is processing
            uint32_t num_input_cols_processed = 0;
            while (num_input_cols_processed < num_unpadded_cols_per_input_block) {
                uint32_t num_cols_to_write = std::min(
                    num_unpadded_cols_per_input_block - num_input_cols_processed,
                    num_cols_remaining_in_current_output_block);
                uint32_t num_bytes_to_write = num_cols_to_write * output_element_size;

                CoreLocalMem<uint32_t> src(current_l1_read_addr);
                noc.async_write(
                    src,
                    s,
                    num_bytes_to_write,
                    {.offset_bytes = 0},
                    {.page_id = output_page_id, .offset_bytes = output_offset_within_page_in_bytes});

                num_input_cols_processed += num_cols_to_write;
                current_l1_read_addr += num_bytes_to_write;
                output_page_id++;
                num_cols_remaining_in_current_output_block = num_cols_per_output_block;
                output_offset_within_page_in_bytes = 0;
            }
        }

        noc.async_write_barrier();
        dfb_out.pop_front(num_tiles_per_input_block);
    };

    // Each input block processed separately
    uint32_t height_wise_input_block_index = height_wise_input_block_start_index;
    for (uint32_t i = 0; i < num_input_blocks_to_process; ++i) {
        write_tiles_in_current_block(height_wise_input_block_index);
        height_wise_input_block_index++;
    }
}
