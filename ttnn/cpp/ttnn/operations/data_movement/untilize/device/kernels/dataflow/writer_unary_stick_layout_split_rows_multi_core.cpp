// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

void kernel_main() {
    // run-time args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);
    const uint32_t block_across_height_start_id = get_arg_val<uint32_t>(2);

    // compile-time args
    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(1);
    constexpr uint32_t stick_size = get_compile_time_arg_val(2);
    constexpr bool stick_size_is_power_of_two = get_compile_time_arg_val(3) == 1;
    constexpr uint32_t log_base_2_of_page_size = get_compile_time_arg_val(4);
    constexpr uint32_t tile_height = get_compile_time_arg_val(5);
    constexpr uint32_t num_tiles_per_block = get_compile_time_arg_val(6);  // TODO: run-time to handle uneven shard
    constexpr uint32_t num_output_columns = get_compile_time_arg_val(7);

    constexpr uint32_t num_columns_of_blocks = get_compile_time_arg_val(8);
    constexpr uint32_t num_blocks_per_column_row = get_compile_time_arg_val(9);
    constexpr uint32_t single_block_width_size = get_compile_time_arg_val(10);

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
        experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(1));
    experimental::ShardedAddrGen<tensor_shard_info> s = {.bank_base_address = dst_addr, .shard_array = mapping_table};
#else
    const auto s = get_interleaved_addr_gen<dst_is_dram, stick_size_is_power_of_two>(
        dst_addr, stick_size, log_base_2_of_page_size);
#endif

    auto write_tiles_in_current_block = [&](uint32_t block_across_height_id) {
        cb_wait_front(cb_id_out0, num_tiles_per_block);

        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        for (uint32_t j = 0; j < tile_height; ++j) {
            uint32_t num_complete_rows_already_processed =
                (block_across_height_id * tile_height + j) * num_output_columns;

            // The input row may need to be broken up into different output columns.
            // And/or we may be writing to the middle of a page.
            for (/*each output column we need to write to*/) {
                uint32_t page_id;
                uint32_t offset_within_page_in_bytes;
                uint32_t num_bytes_to_write;

                uint64_t dst_noc_addr = get_noc_addr(page_id, s, offset_within_page_in_bytes);
                noc_async_write(l1_read_addr, dst_noc_addr, num_bytes_to_write);
                l1_read_addr += num_bytes_to_write;
            }
        }

        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, num_tiles_per_block);
    };

    // Each row of tiles (block) processed separately
    for (uint32_t i = 0; i < num_blocks; ++i) {
        uint32_t block_across_height_id = block_across_height_start_id + i;  // 0-indexed

        // Process the current block
        write_tiles_in_current_block(block_across_height_id);
    }
}
