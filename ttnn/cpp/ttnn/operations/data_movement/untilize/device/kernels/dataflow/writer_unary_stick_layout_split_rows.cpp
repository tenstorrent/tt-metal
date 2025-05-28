// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

void kernel_main() {
    // const-expr variables
    constexpr uint32_t cb_id_out0 = 16;
    constexpr uint32_t tile_height = 32;

    // run-time args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks_across_height = get_arg_val<uint32_t>(1);
    const uint32_t num_columns_of_blocks = get_arg_val<uint32_t>(2);
    const uint32_t num_blocks_per_column_row = get_arg_val<uint32_t>(3);
    const uint32_t num_tiles_per_block = get_arg_val<uint32_t>(4);
    const uint32_t single_block_width_size = get_arg_val<uint32_t>(5);
    const uint32_t stick_size = get_arg_val<uint32_t>(6);

    // compile-time args
    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool stick_size_is_power_of_two = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t log_base_2_of_page_size = get_compile_time_arg_val(2);

#ifdef SHARDED
    using tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(3),   // Memory layout
        get_compile_time_arg_val(4),   // The number of sharding cores
        get_compile_time_arg_val(5),   // The page size we offset each write to
        get_compile_time_arg_val(6),   // The number of pages in each sharding row not including padding pages
        get_compile_time_arg_val(7),   // This defines times when contiguous pages can't be calculated
        get_compile_time_arg_val(8),   // pages_per_shard_x
        get_compile_time_arg_val(9)>;  // pages_per_shard_y

    const auto [mapping_table, rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(7));
    experimental::ShardedAddrGen<tensor_shard_info> s = {.bank_base_address = dst_addr, .shard_array = mapping_table};
#else
    const auto s = get_interleaved_addr_gen<dst_is_dram, stick_size_is_power_of_two>(
        dst_addr, stick_size, log_base_2_of_page_size);
#endif

    uint64_t base_dst_noc_addr[tile_height];

    auto write_tiles_in_current_block = [&]() {
        cb_wait_front(cb_id_out0, num_tiles_per_block);

        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        for (uint32_t l = 0; l < tile_height; ++l) {
            uint64_t dst_noc_addr = base_dst_noc_addr[l];
            noc_async_write(l1_read_addr, dst_noc_addr, single_block_width_size);
            l1_read_addr += single_block_width_size;
            base_dst_noc_addr[l] += single_block_width_size;
        }

        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, num_tiles_per_block);
    };

    // Each row of tiles processed separately
    for (uint32_t i = 0; i < num_blocks_across_height; ++i) {
        // If width or block sharded, we'll have multiple columns of rows/blocks
        for (uint32_t j = 0; j < num_columns_of_blocks; ++j) {
            // Determine the base addresses for the row of blocks in the current column
            for (uint32_t k = 0; k < tile_height; ++k) {
                uint32_t stick_id = (i * 32 + k) * num_columns_of_blocks + j;
                base_dst_noc_addr[k] = get_noc_addr(stick_id, s);
            }

            // For each block in the current column, write all tiles in that block starting at our dst_noc_address
            for (uint32_t k = 0; k < num_blocks_per_column_row; ++k) {
                write_tiles_in_current_block();
            }
        }
    }
}
