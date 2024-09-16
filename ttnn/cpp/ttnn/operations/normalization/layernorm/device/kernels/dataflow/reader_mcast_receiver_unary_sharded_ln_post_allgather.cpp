// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"


// split REDUCE across cores
void kernel_main() {

    uint32_t reduce_receiver_semaphore_addr  = get_semaphore(get_compile_time_arg_val(0));
    uint32_t reduce_sender_semaphore_addr    = get_semaphore(get_compile_time_arg_val(1));
    constexpr uint32_t num_blocks                      = get_compile_time_arg_val(2);
    constexpr uint32_t block_h                         = get_compile_time_arg_val(3);
    const bool is_all_to_all_worker                    = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t num_all_to_all_workers          = get_compile_time_arg_val(5);
    constexpr uint32_t num_tiles_per_worker            = get_compile_time_arg_val(6);
    constexpr uint32_t num_tiles_per_worker_last       = get_compile_time_arg_val(7);
    constexpr bool row_major                           = (bool) get_compile_time_arg_val(8);
    constexpr uint32_t num_x                           = get_compile_time_arg_val(9);
    constexpr uint32_t num_y                           = get_compile_time_arg_val(10);
    constexpr bool use_two_stage_reduce                              = (bool) get_compile_time_arg_val(11);
    constexpr uint32_t num_blocks_first_stage                        = get_compile_time_arg_val(12);
    constexpr uint32_t num_blocks_second_stage                       = get_compile_time_arg_val(13);
    uint32_t reduce_second_stage_semaphore_addr            = get_semaphore(get_compile_time_arg_val(14));

    const bool is_last_all_to_all_worker                = get_arg_val<uint32_t>(0);
    const uint32_t all_to_all_tile_offset_bytes         = get_arg_val<uint32_t>(1);
    const bool is_second_stage_reader                   = get_arg_val<uint32_t>(2);
    const uint32_t start_x                              = get_arg_val<uint32_t>(3);
    const uint32_t start_y                              = get_arg_val<uint32_t>(4);
    volatile tt_l1_ptr uint32_t * in0_remote_noc_x          = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(5));
    volatile tt_l1_ptr uint32_t * in0_remote_noc_y          = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(5 + num_x));

    const uint32_t num_tiles_to_read = is_last_all_to_all_worker ? num_tiles_per_worker_last : num_tiles_per_worker;

    constexpr uint32_t cb_ex_partial = tt::CB::dataflow0; // E[x] partial reduce
    constexpr uint32_t cb_ex = tt::CB::dataflow1; // E[x] global reduce
    constexpr uint32_t cb_ex_external = tt::CB::dataflow2;
    constexpr uint32_t cb_ex_partial2 = tt::CB::dataflow3; // E[(x-E[x])^2] partial reduce
    constexpr uint32_t cb_ex_external2 = tt::CB::dataflow5;
    constexpr uint32_t cb_ex2pe = tt::CB::c_intermed3;
    constexpr uint32_t cb_ex_global = tt::CB::dataflow7; // E[x] global reduce
    constexpr uint32_t cb_ex2_global = tt::CB::dataflow6; // E[x2] global reduce

    const uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial2); // tile size
    const DataFormat data_format = get_dataformat(cb_ex_partial2); // data format

    #ifdef RMSNORM
    constexpr uint32_t stats_tiles = 1;
    #else
    constexpr uint32_t stats_tiles = 2;
    #endif

    uint64_t remote_noc_addrs_first_stage[is_all_to_all_worker ? num_blocks_first_stage : 1];
    uint64_t remote_noc_addrs_second_stage[is_all_to_all_worker ? num_blocks_second_stage : 1];
    if constexpr (is_all_to_all_worker) {
        if constexpr (use_two_stage_reduce) {
            uint32_t x = start_x, y = start_y;
            for (uint32_t i = 0; i < num_blocks_first_stage; ++i) {
                remote_noc_addrs_first_stage[i] = get_noc_addr(in0_remote_noc_x[x], in0_remote_noc_y[y], 0);
                if constexpr(row_major) {
                    ++x;
                    if (x == num_x) {
                        x = 0;
                    }
                } else {
                    ++y;
                    if (y == num_y) {
                        y = 0;
                    }
                }
            }
            if constexpr(row_major) {
                x = start_x;
                y = 0;
            } else {
                x = 0;
                y = start_y;
            }
            for (uint32_t i = 0; i < num_blocks_second_stage; ++i) {
                remote_noc_addrs_second_stage[i] = get_noc_addr(in0_remote_noc_x[x], in0_remote_noc_y[y], 0);
                if constexpr(row_major) {
                    ++y;
                } else {
                    ++x;
                }
            }
        } else {
            uint32_t x = start_x, y = start_y;
            for (uint32_t i = 0; i < num_blocks; ++i) {
                remote_noc_addrs_first_stage[i] = get_noc_addr(in0_remote_noc_x[x], in0_remote_noc_y[y], 0);
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
        }
    } else {
        remote_noc_addrs_first_stage[0] = get_noc_addr(in0_remote_noc_x[0], in0_remote_noc_y[0], 0);
    }

    volatile tt_l1_ptr uint32_t* reduce_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_receiver_semaphore_addr);
    volatile tt_l1_ptr uint32_t* reduce_sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sender_semaphore_addr);
    volatile tt_l1_ptr uint32_t* reduce_second_stage_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_second_stage_semaphore_addr);

    const uint64_t reduce_receiver_semaphore_noc_addr = get_noc_addr(in0_remote_noc_x[0], in0_remote_noc_y[0], reduce_receiver_semaphore_addr);

    // inc mcast sender
    noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);
    // inc remote sem
    cb_reserve_back(cb_ex_global, stats_tiles*block_h);
    noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);
    cb_push_back(cb_ex_global, stats_tiles*block_h);
}
