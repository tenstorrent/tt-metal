// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

void kernel_main() {
    // run-time args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_input_blocks_to_process = get_arg_val<uint32_t>(1);
    const uint32_t height_wise_input_block_start_index = get_arg_val<uint32_t>(2);
    const uint32_t width_wise_input_block_index = get_arg_val<uint32_t>(3);
    const uint32_t num_unpadded_cols_per_input_block = get_arg_val<uint32_t>(4);

    // compile-time args
    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(1);
    constexpr uint32_t output_stick_size = get_compile_time_arg_val(2);
    constexpr bool output_stick_size_is_power_of_two = get_compile_time_arg_val(3) == 1;
    constexpr uint32_t output_log_base_2_of_page_size = get_compile_time_arg_val(4);
    constexpr uint32_t tile_height = get_compile_time_arg_val(5);
    constexpr uint32_t num_tiles_per_input_block = get_compile_time_arg_val(6);
    constexpr uint32_t num_output_blocks_across_width = get_compile_time_arg_val(7);
    constexpr uint32_t output_element_size = get_compile_time_arg_val(8);
    constexpr uint32_t num_cols_per_input_block = get_compile_time_arg_val(9);
    constexpr uint32_t num_cols_per_output_block = get_compile_time_arg_val(10);

#ifdef SHARDED
    using tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(11),   // Memory layout
        get_compile_time_arg_val(12),   // The number of sharding cores
        get_compile_time_arg_val(13),   // The page size we offset each write to
        get_compile_time_arg_val(14),   // The number of pages in each sharding row not including padding pages
        get_compile_time_arg_val(15),   // This defines times when contiguous pages can't be calculated
        get_compile_time_arg_val(16),   // pages_per_shard_x
        get_compile_time_arg_val(17)>;  // pages_per_shard_y

    const auto [mapping_table, rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(5));
    experimental::ShardedAddrGen<tensor_shard_info> s = {.bank_base_address = dst_addr, .shard_array = mapping_table};
#else
    const auto s = get_interleaved_addr_gen<dst_is_dram, output_stick_size_is_power_of_two>(
        dst_addr, output_stick_size, output_log_base_2_of_page_size);
#endif
    // TODO: (GR) num_cols_per_input_block = num_tiles_per_input_block * tile_width

    auto write_tiles_in_current_block = [&](uint32_t block_height_index) {
        cb_wait_front(cb_id_out0, num_tiles_per_input_block);

        // Base address of the row of elements we are going to be writing.
        // If the input is unevenly sharded width wise and we are processing a block in the
        // last shard width wise, we will not be writing the entire row of elements as the
        // last x elements will be garbage. Tracking the base address of the row allows us to
        // easily increment the current read addr to the next row of elements.
        uint32_t base_l1_read_addr = get_read_ptr(cb_id_out0);

        // Process each row of elements in the input block
        for (uint32_t j = 0; j < tile_height; ++j) {
            uint32_t current_l1_read_addr = base_l1_read_addr + j * num_cols_per_input_block * output_element_size;
            uint32_t num_rows_already_processed = block_height_index * tile_height + j;

            // For width or block sharding, the input tensor may have more/less shards width wise compared
            // to the output tensor. In the less case different sections of the row-major row in the input block
            // may map to different output blocks. In both cases, we may be writing to the middle of an output page.
            uint32_t num_input_cols_processed = 0;
            while (num_input_cols_processed < num_unpadded_cols_per_input_block) {
                // Index of the column that we are going to start writing elements to (relative to the entire tensor)
                uint32_t write_col_index =
                    width_wise_input_block_index * num_cols_per_input_block + num_input_cols_processed;

                // page_id to write to
                uint32_t output_page_id_within_row = write_col_index / num_cols_per_output_block;
                uint32_t output_page_id =
                    num_rows_already_processed * num_output_blocks_across_width + output_page_id_within_row;

                // Offset within page we're writing to
                uint32_t num_cols_already_processed_in_output_block = write_col_index % num_cols_per_output_block;
                uint32_t output_offset_within_page_in_bytes =
                    num_cols_already_processed_in_output_block * output_element_size;

                // How many elements to write from the input block to the output block.
                // Min of the number of remaining unprocessed columns in the input block
                // and the number of remaining unprocessed columns in the output block.
                uint32_t num_cols_to_write = std::min(
                    num_unpadded_cols_per_input_block - num_input_cols_processed,
                    num_cols_per_output_block - num_cols_already_processed_in_output_block);
                uint32_t num_bytes_to_write = num_cols_to_write * output_element_size;

                // Perform the write
                uint64_t dst_noc_addr = get_noc_addr(output_page_id, s, output_offset_within_page_in_bytes);
                noc_async_write(current_l1_read_addr, dst_noc_addr, num_bytes_to_write);

                // Update number of columns processed and increment the read ptr
                num_input_cols_processed += num_cols_to_write;
                current_l1_read_addr += num_bytes_to_write;
            }
        }

        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, num_tiles_per_input_block);
    };

    // Each input block processed separately
    for (uint32_t i = 0; i < num_input_blocks_to_process; ++i) {
        uint32_t height_wise_input_block_index = height_wise_input_block_start_index + i;

        // Process the current block
        write_tiles_in_current_block(height_wise_input_block_index);
    }
}
