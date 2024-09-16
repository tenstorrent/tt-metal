// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

// split REDUCE across cores
void kernel_main() {

    uint32_t reduce_receiver_semaphore_addr       = get_semaphore(get_compile_time_arg_val(0));
    uint32_t reduce_sender_semaphore_addr         = get_semaphore(get_compile_time_arg_val(1));
    constexpr uint32_t num_blocks                           = get_compile_time_arg_val(2);
    constexpr uint32_t block_h                              = get_compile_time_arg_val(3);
    constexpr uint32_t block_h_size_bytes                   = get_compile_time_arg_val(4);
    constexpr uint32_t num_all_to_all_workers_first_stage   = get_compile_time_arg_val(5);
    constexpr uint32_t num_tiles_per_worker                 = get_compile_time_arg_val(6);
    constexpr uint32_t num_tiles_per_worker_bytes           = get_compile_time_arg_val(7);
    constexpr uint32_t num_tiles_per_worker_last            = get_compile_time_arg_val(8);
    constexpr uint32_t num_tiles_per_worker_last_bytes      = get_compile_time_arg_val(9);
    constexpr bool row_major                                = (bool) get_compile_time_arg_val(10);
    constexpr uint32_t num_x                                = get_compile_time_arg_val(11);
    constexpr uint32_t num_y                                = get_compile_time_arg_val(12);
    constexpr bool use_two_stage_reduce                     = (bool) get_compile_time_arg_val(13);
    constexpr uint32_t num_blocks_first_stage               = get_compile_time_arg_val(14);
    constexpr uint32_t num_blocks_second_stage              = get_compile_time_arg_val(15);
    uint32_t reduce_second_stage_semaphore_addr   = get_semaphore(get_compile_time_arg_val(16));

    const uint32_t mcast_dest_noc_start_x               = get_arg_val<uint32_t>(0);
    const uint32_t mcast_dest_noc_start_y               = get_arg_val<uint32_t>(1);
    const uint32_t mcast_dest_noc_end_x                 = get_arg_val<uint32_t>(2);
    const uint32_t mcast_dest_noc_end_y                 = get_arg_val<uint32_t>(3);
    const uint32_t start_x                              = get_arg_val<uint32_t>(4);
    const uint32_t start_y                              = get_arg_val<uint32_t>(5);

    tt_l1_ptr uint32_t * in0_remote_noc_x          = (tt_l1_ptr uint32_t*)(get_arg_addr(6));
    tt_l1_ptr uint32_t * in0_remote_noc_y          = (tt_l1_ptr uint32_t*)(get_arg_addr(6 + num_x));

    constexpr uint32_t cb_ex_partial = tt::CB::dataflow0;
    constexpr uint32_t cb_ex = tt::CB::dataflow1;
    constexpr uint32_t cb_ex_external = tt::CB::dataflow2;
    constexpr uint32_t cb_ex_partial2 = tt::CB::dataflow3;
    constexpr uint32_t cb_stats = tt::CB::c_in7;
    constexpr uint32_t cb_stats_reduced = tt::CB::c_intermed4; // E[(x-E[x])^2] global reduce
    constexpr uint32_t cb_ex_external2 = tt::CB::dataflow5;
    constexpr uint32_t cb_ex2pe = tt::CB::c_intermed3;
    constexpr uint32_t cb_ex_global = tt::CB::dataflow7; // E[x] global reduce
    constexpr uint32_t cb_ex2_global = tt::CB::dataflow6; // E[x2] global reduce

    const uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial2);
    const DataFormat data_format = get_dataformat(cb_ex_partial2);

    uint64_t remote_noc_addrs[num_blocks];

    uint32_t x = start_x, y = start_y;
    for (uint32_t i = 0; i < num_blocks; ++i) {
        remote_noc_addrs[i] = get_noc_addr(in0_remote_noc_x[x], in0_remote_noc_y[y], 0);
        if constexpr(row_major) {
            ++x;
            if (x == num_x) {
                x = 0;
                ++y;
                if (y == num_y) {
                    y = 0;
                }
            }
        } else {
            ++y;
            if (y == num_y) {
                y = 0;
                ++x;
                if (x == num_x) {
                    x = 0;
                }
            }
        }
    }

    const uint64_t multicast_data_noc = get_noc_multicast_addr(
        mcast_dest_noc_start_x,
        mcast_dest_noc_start_y,
        mcast_dest_noc_end_x,
        mcast_dest_noc_end_y,
        0);

    const uint64_t reduce_sender_semaphore_noc_addr = multicast_data_noc | reduce_sender_semaphore_addr;
    #ifdef RMSNORM
    constexpr uint32_t stats_tiles = 1;
    #else
    constexpr uint32_t stats_tiles = 2;
    #endif

    volatile tt_l1_ptr uint32_t* reduce_sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sender_semaphore_addr);
    volatile tt_l1_ptr uint32_t* reduce_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_receiver_semaphore_addr);
    volatile tt_l1_ptr uint32_t* reduce_second_stage_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_second_stage_semaphore_addr);
     const auto& global_semaphore_set = [&]() __attribute__((always_inline))
    {
        *reduce_sender_semaphore_addr_ptr = VALID;

        noc_semaphore_set_multicast_loopback_src(
                            reduce_sender_semaphore_addr,
                            reduce_sender_semaphore_noc_addr,
                            num_blocks,
                            false,
                            false);
    };

    const auto& global_reduce_sender = [&](const uint32_t cb_ex, const uint32_t cb_ex_global) __attribute__((always_inline))
    {
        // global reduce


        uint32_t l1_read_addr_ex = get_read_ptr(cb_ex);
        uint32_t l1_read_addr_ex_global = get_read_ptr(cb_ex_global);
        // noc_semaphore_wait(reduce_receiver_semaphore_addr_ptr, num_blocks);
        // noc_semaphore_set(reduce_receiver_semaphore_addr_ptr, 0);
        noc_async_write_multicast_loopback_src(
                            l1_read_addr_ex,
                            multicast_data_noc | l1_read_addr_ex_global,
                            stats_tiles*num_tiles_per_worker_bytes,
                            num_blocks,
                            false,
                            false);
        noc_async_write_barrier();

    };


    cb_wait_front(cb_stats_reduced, stats_tiles*block_h);
    cb_reserve_back(cb_ex_global, block_h);
    global_reduce_sender(cb_stats_reduced, cb_ex_global);
    cb_push_back(cb_ex_global, stats_tiles*block_h);
    cb_pop_front(cb_stats_reduced, stats_tiles*block_h);
    global_semaphore_set();

}
