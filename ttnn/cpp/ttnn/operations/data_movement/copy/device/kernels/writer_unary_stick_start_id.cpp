//  SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t stick_size = get_arg_val<uint32_t>(1);
    uint32_t num_sticks = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);
    uint32_t num_shards = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(0);

    constexpr bool dst0_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool dst_stick_size_is_pow2 = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t dst_log_base_2_of_page_size = get_compile_time_arg_val(3);

    typedef ShardedInfo<
        get_compile_time_arg_val(4),
        get_compile_time_arg_val(5),
        get_compile_time_arg_val(6),
        get_compile_time_arg_val(7),
        get_compile_time_arg_val(8),
        get_compile_time_arg_val(9),
        get_compile_time_arg_val(10)>
        tensor_shard_info;

    const auto [mapping_table, rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(5));
    experimental::ShardedAddrGen<tensor_shard_info> s0 = {.bank_base_address = dst_addr, .shard_array = mapping_table};

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_sticks;
    for (uint32_t i = start_id; i != end_id; --i) {
        for (uint32_t k = num_shards - 1; k >= 0; k--) {
#else
    uint32_t end_id = start_id + num_sticks;
    for (uint32_t i = start_id; i < end_id; ++i) {
        for (uint32_t k = 0; k < num_shards; k++) {
#endif
            uint32_t stick_index = i * num_shards + k;
            cb_wait_front(cb_id_out0, 1);
            uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
            uint64_t dst_noc_addr = get_noc_addr(stick_index, s0);
            noc_async_write(l1_read_addr, dst_noc_addr, stick_size);
            noc_async_write_barrier();
            cb_pop_front(cb_id_out0, 1);
        }
    }
}
