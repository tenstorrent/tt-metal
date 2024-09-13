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

    constexpr uint32_t cb_ex_partial2 = tt::CB::dataflow3; // E[(x-E[x])^2] partial reduce
    constexpr uint32_t cb_ex2 = tt::CB::dataflow4; // E[(x-E[x])^2] global reduce
    constexpr uint32_t cb_ex_external2 = tt::CB::dataflow5;
    constexpr uint32_t cb_ex2pe = tt::CB::c_intermed3;
    constexpr uint32_t cb_ex2_global = tt::CB::dataflow6; // E[x2] global reduce

    const uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial2); // tile size
    const DataFormat data_format = get_dataformat(cb_ex_partial2); // data format

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
        remote_noc_addrs_second_stage[0] = get_noc_addr(in0_remote_noc_x[0], in0_remote_noc_y[0], 0);
    }

    volatile tt_l1_ptr uint32_t* reduce_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_receiver_semaphore_addr);
    volatile tt_l1_ptr uint32_t* reduce_sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sender_semaphore_addr);
    volatile tt_l1_ptr uint32_t* reduce_second_stage_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_second_stage_semaphore_addr);

    const uint64_t reduce_receiver_semaphore_noc_addr = get_noc_addr(in0_remote_noc_x[0], in0_remote_noc_y[0], reduce_receiver_semaphore_addr);
    const uint64_t reduce_second_stage_receiver_semaphore_noc_addr = remote_noc_addrs_second_stage[0] | reduce_second_stage_semaphore_addr;

    const auto& global_reduce_receiver = [&](const uint32_t cb_partial, const uint32_t cb_external, const uint32_t cb_ex, const uint32_t cb_ex_global, const uint32_t cb_reduce_first_stage) __attribute__((always_inline))
    {
        uint32_t num_tiles_per_partial_result = 2;
        #ifdef RMSNORM
        num_tiles_per_partial_result = 1;
        #endif
        // global reduce
        // wait for local data ready
        cb_wait_front(cb_partial, num_tiles_per_partial_result*block_h); // two tiles * block_h


        // inc mcast sender
        noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);
        noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);
        noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);


        if constexpr (is_all_to_all_worker) {
            // read data from other cores - reduce first stage
            uint32_t l1_read_addr_ex_par = get_read_ptr(cb_partial);
            l1_read_addr_ex_par += all_to_all_tile_offset_bytes;
            // read data from other cores - second stage reduce
            uint32_t l1_read_addr_ex = 0;
            uint32_t block_index_stride = 0;
            if constexpr(use_two_stage_reduce) {
                l1_read_addr_ex = get_read_ptr(cb_reduce_first_stage);
                if constexpr(row_major) {
                    block_index_stride = num_x;
                } else {
                    block_index_stride = num_y;
                }
            }
            for (uint32_t i = 0; i < num_tiles_to_read; i++) {
                cb_reserve_back(cb_external, num_tiles_per_partial_result*num_blocks_first_stage);
                uint32_t l1_write_addr_external = get_write_ptr(cb_external);
                for(uint32_t block = 0; block < num_blocks_first_stage; block++) {
                    for(uint32_t tile_idx = 0; tile_idx < num_tiles_per_partial_result; tile_idx++) { // loops over Sum(X), Sum(X2) --> 2x
                        uint64_t noc_addr_ex_par = remote_noc_addrs_first_stage[block] | (l1_read_addr_ex_par + tile_idx * single_tile_size_bytes); // Updating read address for reading SUm(X) and Sum(X2) per core
                        noc_async_read_one_packet(noc_addr_ex_par, l1_write_addr_external, single_tile_size_bytes);
                        l1_write_addr_external+=single_tile_size_bytes;
                    }
                }
                l1_read_addr_ex_par += single_tile_size_bytes;
                noc_async_read_barrier();
                cb_push_back(cb_external, num_tiles_per_partial_result*num_blocks_first_stage);


                // read data from other cores - reduce first stage
                if constexpr(use_two_stage_reduce) {
                    if (is_second_stage_reader) { // gather data from a column of cores (if row major)
                        if (i == 0) {
                            noc_semaphore_wait(reduce_second_stage_semaphore_addr_ptr, num_blocks_second_stage-1);
                            noc_semaphore_set(reduce_second_stage_semaphore_addr_ptr, 0);
                        }
                        // read data from other cores - second stage reduce
                        for(uint32_t block = 0; block < num_blocks_second_stage - 1; ++block) {
                            uint64_t noc_addr_ex = remote_noc_addrs_second_stage[block + 1] | l1_read_addr_ex;
                            noc_async_read_one_packet(noc_addr_ex, l1_write_addr_external, single_tile_size_bytes);
                            l1_write_addr_external += single_tile_size_bytes;
                        }
                        l1_read_addr_ex += single_tile_size_bytes;
                        noc_async_read_barrier();
                        cb_push_back(cb_external, num_blocks_second_stage - 1);
                    }
                }
            }

            // sync with the gather worker
            cb_wait_front(cb_reduce_first_stage, num_tiles_per_partial_result*num_tiles_to_read);
            noc_semaphore_inc(reduce_second_stage_receiver_semaphore_noc_addr, 1);
        }

    };
    global_reduce_receiver(cb_ex_partial2, cb_ex_external2, cb_ex2, cb_ex2_global, cb_ex2);
}
