// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

void kernel_main() {
    // run-time args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t stick_size = get_arg_val<uint32_t>(1);
    const uint32_t block_height = get_arg_val<uint32_t>(2);
    const uint32_t block_width_bytes = get_arg_val<uint32_t>(3);
    const uint32_t padded_block_width_bytes = get_arg_val<uint32_t>(4);
    const uint32_t input_width_offset_bytes = get_arg_val<uint32_t>(5);
    const uint32_t start_id = get_arg_val<uint32_t>(6);

    // compile-time args
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(0);

    constexpr bool dst0_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool dst_stick_size_is_pow2 = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t dst_log_base_2_of_page_size = get_compile_time_arg_val(3);

    // TODO: (GR) Update indexing
    using tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(1),   // Memory layout
        get_compile_time_arg_val(2),   // The number of sharding cores
        get_compile_time_arg_val(3),   // The page size we offset each write to
        get_compile_time_arg_val(4),   // The number of pages in each sharding row not including padding pages
        get_compile_time_arg_val(5),   // This defines times when contiguous pages can't be calculated
        get_compile_time_arg_val(6),   // pages_per_shard_x
        get_compile_time_arg_val(7)>;  // pages_per_shard_y

    // TODO: (GR) Update indexing
    const auto [mapping_table, rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(7));
    experimental::ShardedAddrGen<tensor_shard_info> s = {
        .bank_base_address = dst_addr + input_width_offset_bytes, .shard_array = mapping_table};

    uint32_t stick_id = start_id;
    cb_wait_front(cb_id_out0, block_height);
    uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
    for (uint32_t h = 0; h < block_height; ++h) {
        uint64_t dst_noc_addr = get_noc_addr(stick_id, s0);
        noc_async_write(l1_read_addr, dst_noc_addr, block_width_bytes);
        stick_id++;
        l1_read_addr += padded_block_width_bytes;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_id_out0, block_height);
}
