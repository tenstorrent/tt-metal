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
    constexpr uint32_t block_ht = get_compile_time_arg_val(3);
    constexpr bool row_major = (bool)get_compile_time_arg_val(8);
    constexpr uint32_t num_cores_x_mcast = get_compile_time_arg_val(9);
    constexpr uint32_t num_cores_y_mcast = get_compile_time_arg_val(10);
    constexpr uint32_t num_bytes_copy_to_combine = get_compile_time_arg_val(15);
    constexpr uint32_t num_combine_tiles_needed = get_compile_time_arg_val(16);

    const bool is_last_all_to_all_worker = get_arg_val<uint32_t>(0);
    const uint32_t all_to_all_tile_offset_bytes = get_arg_val<uint32_t>(1);
    const bool is_second_stage_reader = get_arg_val<uint32_t>(2);
    const uint32_t start_x = get_arg_val<uint32_t>(3);
    const uint32_t start_y = get_arg_val<uint32_t>(4);
    volatile tt_l1_ptr uint32_t* in0_remote_noc_x = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(5));
    volatile tt_l1_ptr uint32_t* in0_remote_noc_y = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(5 + num_x));

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

    // wait for local partial data ready
    cb_wait_front(cb_ex_partial, block_ht);
    cb_wait_front(cb_varx_partial, block_ht);

    // Sync with sender
    noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);
    noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);
    noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);

    // Copy local partial E[x] and Var[x] data to combine buffers
    cb_reserve_back(cb_ex_combine, num_combine_tiles_needed);
    cb_reserve_back(cb_varx_combine, num_combine_tiles_needed);
    auto l1_read_addr_ex_par = get_read_ptr(cb_ex_partial);
    auto l1_read_addr_varx_par = get_read_ptr(cb_varx_partial);
    auto l1_write_addr_ex_combine = get_write_ptr(cb_ex_combine);
    auto l1_write_addr_varx_combine = get_write_ptr(cb_varx_combine);
    for (uint32_t i = 0; i < block_ht; i++) {
        // Copy partial E[x]
        noc_async_read(l1_read_addr_ex_par, l1_write_addr_ex_combine, num_bytes_copy_to_combine);
        l1_read_addr_ex_par += single_tile_size_bytes;
        l1_write_addr_ex_combine += num_bytes_copy_to_combine;

        // Copy partial Var[x]
        noc_async_read(l1_read_addr_varx_par, l1_write_addr_varx_combine, num_bytes_copy_to_combine);
        l1_read_addr_varx_par += single_tile_size_bytes;
        l1_write_addr_varx_combine += num_bytes_copy_to_combine;
    }
    noc_async_read_barrier();
    // TODO RM: Do we need to push back if we're syncing via semaphores?
    // cb_push_back(cb_ex_combine, num_combine_tiles_needed);
    // cb_push_back(cb_varx_combine, num_combine_tiles_needed);

    cb_pop_front(cb_ex_partial, block_ht);
    cb_pop_front(cb_varx_partial, block_ht);

    // Signal to sender that combine buffers are ready
    noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);
    noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);
    noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);

    // Make space for the global results
    cb_reserve_back(cb_ex_global, block_ht);
    cb_reserve_back(cb_varx_global, block_ht);

    // Read the global combine results from sender when sender
    // signals that it is ready
    noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);
    cb_push_back(cb_ex_global, block_ht);
    cb_push_back(cb_varx_global, block_ht);
}
