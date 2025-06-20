// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

void kernel_main() {
    // run-time args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);

    // compile-time args
    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(1);
    constexpr uint32_t output_stick_size = get_compile_time_arg_val(2);
    constexpr bool output_stick_size_is_power_of_two = get_compile_time_arg_val(3) == 1;
    constexpr uint32_t output_log_base_2_of_page_size = get_compile_time_arg_val(4);
    constexpr uint32_t tile_height = get_compile_time_arg_val(5);
    constexpr uint32_t num_blocks_across_height = get_compile_time_arg_val(6);
    constexpr uint32_t num_output_columns_of_blocks = get_compile_time_arg_val(7);
    constexpr uint32_t num_blocks_per_output_column_row = get_compile_time_arg_val(8);
    constexpr uint32_t num_tiles_per_output_block = get_compile_time_arg_val(9);
    constexpr uint32_t output_single_block_width_size = get_compile_time_arg_val(10);

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
    const auto s = get_interleaved_addr_gen<dst_is_dram, output_stick_size_is_power_of_two>(
        dst_addr, output_stick_size, output_log_base_2_of_page_size);
#endif

    uint64_t base_dst_noc_addr[tile_height];

    auto write_tiles_in_current_block = [&]() {
        cb_wait_front(cb_id_out0, num_tiles_per_output_block);

        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        for (uint32_t l = 0; l < tile_height; ++l) {
            uint64_t dst_noc_addr = base_dst_noc_addr[l];
            noc_async_write(l1_read_addr, dst_noc_addr, output_single_block_width_size);
            l1_read_addr += output_single_block_width_size;
            base_dst_noc_addr[l] += output_single_block_width_size;
        }

        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, num_tiles_per_output_block);
    };

    // Each row of tiles processed separately
    for (uint32_t i = 0; i < num_blocks_across_height; ++i) {
        // If width or block sharded, we'll have multiple columns of rows/blocks
        for (uint32_t j = 0; j < num_output_columns_of_blocks; ++j) {
            // Determine the base addresses for the row of blocks in the current column
            for (uint32_t k = 0; k < tile_height; ++k) {
                uint32_t num_complete_rows_already_processed = (i * tile_height + k) * num_output_columns_of_blocks;
                uint32_t stick_id = num_complete_rows_already_processed + j;
                base_dst_noc_addr[k] = get_noc_addr(stick_id, s);
            }

            for (uint32_t k = 0; k < num_blocks_per_output_column_row; ++k) {
                write_tiles_in_current_block();
            }
        }
    }
}
