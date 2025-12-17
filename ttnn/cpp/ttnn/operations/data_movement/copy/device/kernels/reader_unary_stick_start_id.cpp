// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t stick_size = get_arg_val<uint32_t>(1);
    uint32_t num_sticks = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);
    uint32_t num_shards = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);

    typedef ShardedInfo<
        get_compile_time_arg_val(2),
        get_compile_time_arg_val(3),
        get_compile_time_arg_val(4),
        get_compile_time_arg_val(5),
        get_compile_time_arg_val(6),
        get_compile_time_arg_val(7),
        get_compile_time_arg_val(8)>
        tensor_shard_info;

    const auto [mapping_table, rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(5));
    experimental::ShardedAddrGen<tensor_shard_info> s0 = {.bank_base_address = src_addr, .shard_array = mapping_table};

    // Transaction ID pipelining strategy:
    // Use 2 transaction IDs to overlap NOC reads with computation.
    // This enables pipelined execution, where while one transaction is being processed,
    // the next transaction's NOC read can be issued, improving throughput.
    constexpr uint32_t trid_base = 1;
    constexpr uint32_t num_of_trids = 2;
    auto cb_interface = get_local_cb_interface(cb_id_in0);
    uint32_t aligned_page_size = cb_interface.fifo_page_size;
    uint32_t num_of_transactions = num_sticks * num_shards;

    // Calculate stick indices based on direction
    auto get_stick_index = [&](uint32_t transaction_idx) -> uint32_t {
#ifdef BACKWARDS
        uint32_t i = start_id - (transaction_idx / num_shards);
        uint32_t k = num_shards - 1 - (transaction_idx % num_shards);
#else
        uint32_t i = start_id + (transaction_idx / num_shards);
        uint32_t k = transaction_idx % num_shards;
#endif
        return i * num_shards + k;
    };

    cb_reserve_back(cb_id_in0, num_of_trids);
    uint32_t base_addr = get_write_ptr(cb_id_in0);
    uint32_t write_addrs[num_of_trids];

    // Initial pipeline fill
    for (uint32_t trid = 0; trid < num_of_trids; trid++) {
        write_addrs[trid] = base_addr + trid * aligned_page_size;

        noc_async_read_set_trid(trid_base + trid);
        uint32_t stick_index = get_stick_index(trid);
        uint64_t src_noc_addr = get_noc_addr(stick_index, s0);
        noc_async_read(src_noc_addr, write_addrs[trid], stick_size);
    }

    // Pipelined execution
    uint32_t trid_to_wait = 0;
    uint32_t trid_to_issue = 0;
    for (uint32_t i = 0; i < num_of_transactions; i++) {
        noc_async_read_barrier_with_trid(trid_base + trid_to_wait);
        cb_push_back(cb_id_in0, 1);

        trid_to_wait = (trid_to_wait + 1) % num_of_trids;

        if (i < num_of_transactions - num_of_trids) {
            noc_async_read_set_trid(trid_base + trid_to_issue);
            uint32_t stick_index = get_stick_index(i + num_of_trids);
            uint64_t src_noc_addr = get_noc_addr(stick_index, s0);

            cb_reserve_back(cb_id_in0, num_of_trids);
            noc_async_read(src_noc_addr, write_addrs[trid_to_issue], stick_size);
            trid_to_issue = (trid_to_issue + 1) % num_of_trids;
        }
    }
    noc_async_read_set_trid(0);
}
