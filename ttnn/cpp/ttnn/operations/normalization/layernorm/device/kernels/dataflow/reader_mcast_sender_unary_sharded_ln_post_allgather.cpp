// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

// split REDUCE across cores
void kernel_main() {
    uint32_t reduce_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(1));
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t block_h = get_compile_time_arg_val(3);
    constexpr uint32_t block_h_size_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t num_tiles_per_worker = get_compile_time_arg_val(6);
    constexpr uint32_t num_tiles_per_worker_bytes = get_compile_time_arg_val(7);

    const uint32_t mcast_dest_noc_start_x = get_arg_val<uint32_t>(0);
    const uint32_t mcast_dest_noc_start_y = get_arg_val<uint32_t>(1);
    const uint32_t mcast_dest_noc_end_x = get_arg_val<uint32_t>(2);
    const uint32_t mcast_dest_noc_end_y = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_stats_reduced = tt::CB::c_intermed4;  // [E[x], E[x^2]] local to sender
    constexpr uint32_t cb_ex_global = tt::CB::dataflow7;        // [E[x], E[X^2]] global to all cores

    const uint64_t multicast_data_noc = get_noc_multicast_addr(
        mcast_dest_noc_start_x, mcast_dest_noc_start_y, mcast_dest_noc_end_x, mcast_dest_noc_end_y, 0);

    const uint64_t reduce_sender_semaphore_noc_addr = multicast_data_noc | reduce_sender_semaphore_addr;
#ifdef RMSNORM
    constexpr uint32_t stats_tiles = 1;
#else
    constexpr uint32_t stats_tiles = 2;
#endif

    volatile tt_l1_ptr uint32_t* reduce_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sender_semaphore_addr);
    const auto& global_semaphore_set = [&]() __attribute__((always_inline)) {
        *reduce_sender_semaphore_addr_ptr = VALID;

        noc_semaphore_set_multicast_loopback_src(
            reduce_sender_semaphore_addr, reduce_sender_semaphore_noc_addr, num_blocks, false, false);
    };

    const auto& global_reduce_sender = [&](const uint32_t cb_ex, const uint32_t cb_ex_global)
        __attribute__((always_inline)) {
        uint32_t l1_read_addr_ex = get_read_ptr(cb_ex);
        uint32_t l1_read_addr_ex_global = get_read_ptr(cb_ex_global);
        noc_async_write_multicast_loopback_src(l1_read_addr_ex,
                                               multicast_data_noc | l1_read_addr_ex_global,
                                               stats_tiles * num_tiles_per_worker_bytes,
                                               num_blocks,
                                               false,
                                               false);
        noc_async_write_barrier();
    };

    cb_wait_front(cb_stats_reduced, stats_tiles * block_h);
    cb_reserve_back(cb_ex_global, block_h);
    global_reduce_sender(cb_stats_reduced, cb_ex_global);
    cb_push_back(cb_ex_global, stats_tiles * block_h);
    cb_pop_front(cb_stats_reduced, stats_tiles * block_h);
    global_semaphore_set();
}
