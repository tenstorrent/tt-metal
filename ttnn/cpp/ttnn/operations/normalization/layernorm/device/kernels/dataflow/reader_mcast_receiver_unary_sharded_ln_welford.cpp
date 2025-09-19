// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

// split REDUCE across cores
void kernel_main() {
    uint32_t reduce_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(0));
    uint32_t reduce_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(1));
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t block_h = get_compile_time_arg_val(3);
    const bool is_all_to_all_worker = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t num_all_to_all_workers = get_compile_time_arg_val(5);
    constexpr uint32_t num_tiles_per_worker = get_compile_time_arg_val(6);
    constexpr uint32_t num_tiles_per_worker_last = get_compile_time_arg_val(7);
    constexpr bool row_major = (bool)get_compile_time_arg_val(8);
    constexpr uint32_t num_x = get_compile_time_arg_val(9);
    constexpr uint32_t num_y = get_compile_time_arg_val(10);
    constexpr bool use_two_stage_reduce = (bool)get_compile_time_arg_val(11);
    constexpr uint32_t num_blocks_first_stage = get_compile_time_arg_val(12);
    constexpr uint32_t num_blocks_second_stage = get_compile_time_arg_val(13);
    uint32_t reduce_second_stage_semaphore_addr = get_semaphore(get_compile_time_arg_val(14));
    constexpr uint32_t num_bytes_copy_to_combine = get_compile_time_arg_val(15);

    const bool is_last_all_to_all_worker = get_arg_val<uint32_t>(0);
    const uint32_t all_to_all_tile_offset_bytes = get_arg_val<uint32_t>(1);
    const bool is_second_stage_reader = get_arg_val<uint32_t>(2);
    const uint32_t start_x = get_arg_val<uint32_t>(3);
    const uint32_t start_y = get_arg_val<uint32_t>(4);
    volatile tt_l1_ptr uint32_t* in0_remote_noc_x = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(5));
    volatile tt_l1_ptr uint32_t* in0_remote_noc_y = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(5 + num_x));

    const uint32_t num_tiles_to_read = is_last_all_to_all_worker ? num_tiles_per_worker_last : num_tiles_per_worker;

    constexpr uint32_t cb_ex_partial = tt::CBIndex::c_8;     // E[x] partial result
    constexpr uint32_t cb_ex_combine = tt::CBIndex::c_9;     // E[x] buffer for global combine
    constexpr uint32_t cb_varx_partial = tt::CBIndex::c_11;  // Var[x] partial result
    constexpr uint32_t cb_varx_combine = tt::CBIndex::c_12;  // Var[x] buffer for global combine
    constexpr uint32_t cb_ex_global = tt::CBIndex::c_15;     // Final global E[x]
    constexpr uint32_t cb_varx_global = tt::CBIndex::c_10;   // Final global Var[x]
    constexpr uint32_t cb_ex2pe = tt::CBIndex::c_20;

    const uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial);  // tile size

    uint64_t remote_noc_addrs_first_stage[is_all_to_all_worker ? num_blocks_first_stage : 1];
    uint64_t remote_noc_addrs_second_stage[is_all_to_all_worker ? num_blocks_second_stage : 1];
    if constexpr (is_all_to_all_worker) {
        if constexpr (use_two_stage_reduce) {
            uint32_t x = start_x, y = start_y;
            for (uint32_t i = 0; i < num_blocks_first_stage; ++i) {
                remote_noc_addrs_first_stage[i] = get_noc_addr(in0_remote_noc_x[x], in0_remote_noc_y[y], 0);
                if constexpr (row_major) {
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
            if constexpr (row_major) {
                x = start_x;
                y = 0;
            } else {
                x = 0;
                y = start_y;
            }
            for (uint32_t i = 0; i < num_blocks_second_stage; ++i) {
                remote_noc_addrs_second_stage[i] = get_noc_addr(in0_remote_noc_x[x], in0_remote_noc_y[y], 0);
                if constexpr (row_major) {
                    ++y;
                } else {
                    ++x;
                }
            }
        } else {
            uint32_t x = start_x, y = start_y;
            for (uint32_t i = 0; i < num_blocks; ++i) {
                remote_noc_addrs_first_stage[i] = get_noc_addr(in0_remote_noc_x[x], in0_remote_noc_y[y], 0);
                if constexpr (row_major) {
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

    volatile tt_l1_ptr uint32_t* reduce_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_receiver_semaphore_addr);
    volatile tt_l1_ptr uint32_t* reduce_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sender_semaphore_addr);
    volatile tt_l1_ptr uint32_t* reduce_second_stage_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_second_stage_semaphore_addr);

    const uint64_t reduce_receiver_semaphore_noc_addr =
        get_noc_addr(in0_remote_noc_x[0], in0_remote_noc_y[0], reduce_receiver_semaphore_addr);
    const uint64_t reduce_second_stage_receiver_semaphore_noc_addr =
        remote_noc_addrs_second_stage[0] | reduce_second_stage_semaphore_addr;

    // wait for local partial E[x]data ready
    cb_wait_front(cb_ex_partial, block_h);

    // Sync with sender
    // Invalidate semaphore so receiver waits
    noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);
    // Sync with sender
    noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);
    // Wait for sender to signal readiness
    noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);

    if constexpr (is_all_to_all_worker) {
        // Copy partial E[x] and Var[x] data to combine buffers
        cb_reserve_back(cb_ex_combine, num_combine_tiles_needed);
        cb_reserve_back(cb_varx_combine, num_combine_tiles_needed);
        auto l1_read_addr_ex_par = get_read_ptr(cb_ex_partial);
        l1_read_addr_ex_par += all_to_all_tile_offset_bytes;
        auto l1_read_addr_varx_par = get_read_ptr(cb_varx_partial);
        l1_read_addr_varx_par += all_to_all_tile_offset_bytes;
        auto l1_write_addr_ex_combine = get_write_ptr(cb_ex_combine);
        auto l1_write_addr_varx_combine = get_write_ptr(cb_varx_combine);
        for (uint32_t i = 0; i < num_tiles_to_read; i++) {
            // Copy partial E[x]
            noc_async_read_one_packet(l1_read_addr_ex_par, l1_write_addr_ex_combine, num_bytes_copy_to_combine);
            l1_read_addr_ex_par += single_tile_size_bytes;
            l1_write_addr_ex_combine += num_bytes_copy_to_combine;

            // Copy partial Var[x]
            noc_async_read_one_packet(l1_read_addr_varx_par, l1_write_addr_varx_combine, num_bytes_copy_to_combine);
            l1_read_addr_varx_par += single_tile_size_bytes;
            l1_write_addr_varx_combine += num_bytes_copy_to_combine;
        }
        noc_async_read_barrier();
        cb_push_back(cb_ex_combine, num_combine_tiles_needed);
        cb_push_back(cb_varx_combine, num_combine_tiles_needed);

        // Signal to sender that combine results are ready and wait for sender to complete
        noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);
        noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);
        noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);

        // read data from other cores - second stage reduce
        uint32_t l1_read_addr_ex = 0;
        for (uint32_t i = 0; i < num_tiles_to_read; i++) {
            cb_reserve_back(cb_external, num_blocks_first_stage);
            uint32_t l1_write_addr_external = get_write_ptr(cb_external);
            for (uint32_t block = 0; block < num_blocks_first_stage; block++) {
                uint64_t noc_addr_ex_par = remote_noc_addrs_first_stage[block] | l1_read_addr_ex_par;
                noc_async_read_one_packet(noc_addr_ex_par, l1_write_addr_external, single_tile_size_bytes);
                l1_write_addr_external += single_tile_size_bytes;
            }
            l1_read_addr_ex_par += single_tile_size_bytes;
            noc_async_read_barrier();
            cb_push_back(cb_external, num_blocks_first_stage);

            // read data from other cores - reduce first stage
            if constexpr (use_two_stage_reduce) {
                if (is_second_stage_reader) {  // gather data from a column of cores (if row major)
                    if (i == 0) {
                        noc_semaphore_wait(reduce_second_stage_semaphore_addr_ptr, num_blocks_second_stage - 1);
                        noc_semaphore_set(reduce_second_stage_semaphore_addr_ptr, 0);
                    }

                    // read data from other cores - second stage reduce
                    cb_reserve_back(cb_external, num_blocks_second_stage - 1);
                    for (uint32_t block = 0; block < num_blocks_second_stage - 1; ++block) {
                        uint64_t noc_addr_ex = remote_noc_addrs_second_stage[block + 1] | l1_read_addr_ex;
                        noc_async_read_one_packet(noc_addr_ex, l1_write_addr_external, single_tile_size_bytes);
                        l1_write_addr_external += single_tile_size_bytes;
                    }
                    l1_read_addr_ex += single_tile_size_bytes;
                    noc_async_read_barrier();
                    cb_push_back(cb_external, num_blocks_second_stage - 1);
                } else {
                    cb_reserve_back(cb_external, num_blocks_second_stage - 1);
                    cb_push_back(cb_external, num_blocks_second_stage - 1);
                }
            }
        }

        // read data from other cores - reduce first stage
        if constexpr (use_two_stage_reduce) {
            if (is_second_stage_reader) {  // gather data from a column of cores (if row major)
                // sync with the mcast sender
                cb_wait_front(cb_ex, num_tiles_to_read);
                noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);
            } else {
                // sync with the gather worker
                cb_wait_front(cb_reduce_first_stage, num_tiles_to_read);
                noc_semaphore_inc(reduce_second_stage_receiver_semaphore_noc_addr, 1);
            }
        } else {
            // send result to other cores
            cb_wait_front(cb_ex, num_tiles_to_read);
            noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);
        }
    }

    for (uint32_t block = 0; block < num_all_to_all_workers; ++block) {
        uint32_t num_tiles = block == num_all_to_all_workers - 1 ? num_tiles_per_worker_last : num_tiles_per_worker;
        cb_reserve_back(cb_ex_global, num_tiles);
        noc_semaphore_wait_min(reduce_sender_semaphore_addr_ptr, block + 2);
        cb_push_back(cb_ex_global, num_tiles);
    }
}
