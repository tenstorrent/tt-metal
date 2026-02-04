// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

void kernel_main() {
    // run-time args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_input_blocks_to_process = get_arg_val<uint32_t>(1);
    const uint32_t height_wise_input_block_start_index = get_arg_val<uint32_t>(2);
    const uint32_t num_unpadded_cols_per_input_block = get_arg_val<uint32_t>(3);
    const uint32_t width_wise_output_block_start_index = get_arg_val<uint32_t>(4);
    const uint32_t num_cols_already_processed_in_first_output_block = get_arg_val<uint32_t>(5);

    // compile-time args
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(0);
    constexpr uint32_t output_stick_size = get_compile_time_arg_val(1);
    constexpr uint32_t tile_height = get_compile_time_arg_val(2);
    constexpr uint32_t num_tiles_per_input_block = get_compile_time_arg_val(3);
    constexpr uint32_t num_output_blocks_across_width = get_compile_time_arg_val(4);
    constexpr uint32_t output_element_size = get_compile_time_arg_val(5);
    constexpr uint32_t num_cols_per_input_block = get_compile_time_arg_val(6);
    constexpr uint32_t num_cols_per_output_block = get_compile_time_arg_val(7);
    constexpr auto dst_args = TensorAccessorArgs<8>();
    const auto s = TensorAccessor(dst_args, dst_addr, output_stick_size);

    auto write_tiles_in_current_block = [&](uint32_t block_height_index) {
        cb_wait_front(cb_id_out0, num_tiles_per_input_block);

        // Base address of the row of elements we are going to be writing.
        // If the input is unevenly sharded width wise and we are processing a block in the
        // last shard width wise, we will not be writing the entire row of elements as the
        // last x elements will be garbage. Tracking the base address of the row allows us to
        // easily increment the current_read_addr to the next row of elements.
        uint32_t base_l1_read_addr = get_read_ptr(cb_id_out0);

        // Process each row of elements in the input block
        for (uint32_t j = 0; j < tile_height; ++j) {
            uint32_t current_l1_read_addr = base_l1_read_addr + j * num_cols_per_input_block * output_element_size;

            // Note: For width or block sharding (either input, output, or both), there may be more/less blocks width
            // wise in the output tensor compared to the input tensor. As a result, for the first output block we write
            // to, we may be writing to the middle of a page (i.e. some byte offset within the page). For all following
            // writes we'll be writing to the beginning of a page and not require an offset.

            // Output page_id that we are going to be writing to
            uint32_t num_rows_already_processed = block_height_index * tile_height + j;
            uint32_t num_pages_already_processed_in_previous_rows =
                num_rows_already_processed * num_output_blocks_across_width;
            uint32_t output_page_id =
                num_pages_already_processed_in_previous_rows + width_wise_output_block_start_index;

            // For the first output page that we write to in the current row, it's first x columns may have been
            // written to from a different input block. So we need to calculate the offset within this first output page
            // to write to (which may be 0). We also need to determine how many columns in this output page have not yet
            // been written to in order to determine how many columns from the input block to writer (min of all
            // remaining columns in the input block, and the remaining unprocessed columns in the current output block).
            uint32_t num_cols_remaining_in_current_output_block =
                num_cols_per_output_block - num_cols_already_processed_in_first_output_block;
            uint32_t output_offset_within_page_in_bytes =
                num_cols_already_processed_in_first_output_block * output_element_size;

            // Iterate through all columns in the input block that this core is processing
            uint32_t num_input_cols_processed = 0;
            while (num_input_cols_processed < num_unpadded_cols_per_input_block) {
                // How many elements to write from the input block to the output block.
                // Min of the number of remaining unprocessed columns in the input block
                // and the number of remaining unprocessed columns in the current output block.
                uint32_t num_cols_to_write = std::min(
                    num_unpadded_cols_per_input_block - num_input_cols_processed,
                    num_cols_remaining_in_current_output_block);
                uint32_t num_bytes_to_write = num_cols_to_write * output_element_size;

                // Perform the write
                uint64_t dst_noc_addr = s.get_noc_addr(output_page_id, output_offset_within_page_in_bytes);
                noc_async_write(current_l1_read_addr, dst_noc_addr, num_bytes_to_write);

                // Increment the number of cols we've processed in the input block
                num_input_cols_processed += num_cols_to_write;

                // Increment l1_read_addr past the bytes we just read
                current_l1_read_addr += num_bytes_to_write;

                // Increment the output_page_id we're writing to. If we wrote the entire input block
                // then this has no effect as the while loop terminates. If we wrote to a subset of the
                // input block, then that subset corresponds to an entire output block, so we increment
                // the output_page_id to the following output block.
                output_page_id++;

                // Only the first output block we write to can have some of it's columns already processed/written-to
                num_cols_remaining_in_current_output_block = num_cols_per_output_block;
                output_offset_within_page_in_bytes = 0;
            }
        }

        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, num_tiles_per_input_block);
    };

    // Each input block processed separately
    uint32_t height_wise_input_block_index = height_wise_input_block_start_index;
    for (uint32_t i = 0; i < num_input_blocks_to_process; ++i) {
        // Process the current block
        write_tiles_in_current_block(height_wise_input_block_index);
        height_wise_input_block_index++;
    }
}
