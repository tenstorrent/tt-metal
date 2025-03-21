// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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
    const bool is_all_to_all_worker = get_compile_time_arg_val(3) == 1;
    constexpr bool row_major = (bool)get_compile_time_arg_val(4);
    constexpr uint32_t num_x = get_compile_time_arg_val(5);
    constexpr uint32_t num_y = get_compile_time_arg_val(6);
    constexpr bool use_two_stage_reduce = (bool)get_compile_time_arg_val(7);
    constexpr uint32_t num_blocks_first_stage = get_compile_time_arg_val(8);
    constexpr uint32_t num_blocks_second_stage = get_compile_time_arg_val(9);
    uint32_t reduce_second_stage_semaphore_addr = get_semaphore(get_compile_time_arg_val(10));

    const uint32_t all_to_all_tile_offset_bytes = get_arg_val<uint32_t>(0);
    const bool is_second_stage_reader = get_arg_val<uint32_t>(1);
    const uint32_t start_x = get_arg_val<uint32_t>(2);
    const uint32_t start_y = get_arg_val<uint32_t>(3);
    volatile tt_l1_ptr uint32_t* in0_remote_noc_x = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(4));
    volatile tt_l1_ptr uint32_t* in0_remote_noc_y = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(4 + num_x));

    constexpr uint32_t cb_ex_partial2 = tt::CBIndex::c_2;  // E[(x-E[x])^2] partial reduce
    constexpr uint32_t cb_ex2 = tt::CBIndex::c_3;          // E[(x-E[x])^2] global reduce
    constexpr uint32_t cb_ex_external2 = tt::CBIndex::c_5;

    const uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial2);  // tile size
    const DataFormat data_format = get_dataformat(cb_ex_partial2);          // data format

    uint64_t remote_noc_addrs_first_stage[is_all_to_all_worker ? num_blocks_first_stage : 1];
    uint64_t remote_noc_addrs_second_stage[is_all_to_all_worker ? num_blocks_second_stage : 1];
    if constexpr (is_all_to_all_worker) {
        if constexpr (use_two_stage_reduce) {
            uint32_t x = start_x, y = start_y;
            for (uint32_t i = 0; i < num_blocks_first_stage; ++i) {
                remote_noc_addrs_first_stage[i] = get_noc_addr(in0_remote_noc_x[x], in0_remote_noc_y[y], 0);
                ++x;
                if (x == num_x) {
                    x = 0;
                }
            }
            x = start_x;
            y = 0;
            for (uint32_t i = 0; i < num_blocks_second_stage; ++i) {
                remote_noc_addrs_second_stage[i] = get_noc_addr(in0_remote_noc_x[x], in0_remote_noc_y[y], 0);
                ++y;
            }
        } else {
            uint32_t x = start_x, y = start_y;
            for (uint32_t i = 0; i < num_blocks; ++i) {
                remote_noc_addrs_first_stage[i] = get_noc_addr(in0_remote_noc_x[x], in0_remote_noc_y[y], 0);
                ++x;
                if (x == num_x) {
                    x = 0;
                    ++y;
                    if (y == num_y) {
                        y = 0;
                    }
                }
            }
        }
    } else {
        remote_noc_addrs_first_stage[0] = get_noc_addr(in0_remote_noc_x[0], in0_remote_noc_y[0], 0);
        remote_noc_addrs_second_stage[0] = get_noc_addr(in0_remote_noc_x[0], in0_remote_noc_y[0], 0);
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

    const auto& global_reduce_receiver = [&](const uint32_t cb_partial,
                                             const uint32_t cb_external,
                                             const uint32_t cb_reduce_first_stage) __attribute__((always_inline)) {
        // global reduce
        // wait for local data ready
        cb_wait_front(cb_partial, 1);

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
            if constexpr (use_two_stage_reduce) {
                l1_read_addr_ex = get_read_ptr(cb_reduce_first_stage);
                block_index_stride = num_x;
            }
            cb_reserve_back(cb_external, num_blocks_first_stage);
            uint32_t l1_write_addr_external = get_write_ptr(cb_external);
            for (uint32_t block = 0; block < num_blocks_first_stage; block++) {
                uint64_t noc_addr_ex_par =
                    remote_noc_addrs_first_stage[block] | (l1_read_addr_ex_par);  // Updating read address for reading
                                                                                  // SUm(X) and Sum(X2) per core
                noc_async_read_one_packet(noc_addr_ex_par, l1_write_addr_external, single_tile_size_bytes);
                l1_write_addr_external += single_tile_size_bytes;
            }
            l1_read_addr_ex_par += single_tile_size_bytes;
            noc_async_read_barrier();
            cb_push_back(cb_external, num_blocks_first_stage);

            // read data from other cores - reduce first stage
            if constexpr (use_two_stage_reduce) {
                if (is_second_stage_reader) {  // gather data from a column of cores (if row major)
                    noc_semaphore_wait(reduce_second_stage_semaphore_addr_ptr, num_blocks_second_stage - 1);
                    noc_semaphore_set(reduce_second_stage_semaphore_addr_ptr, 0);
                    // read data from other cores - second stage reduce
                    for (uint32_t block = 0; block < num_blocks_second_stage - 1; ++block) {
                        uint64_t noc_addr_ex = remote_noc_addrs_second_stage[block + 1] | l1_read_addr_ex;
                        noc_async_read_one_packet(noc_addr_ex, l1_write_addr_external, single_tile_size_bytes);
                        l1_write_addr_external += single_tile_size_bytes;
                    }
                    l1_read_addr_ex += single_tile_size_bytes;
                    noc_async_read_barrier();
                    cb_push_back(cb_external, num_blocks_second_stage - 1);
                }
            }

            // sync with the gather worker
            cb_wait_front(cb_reduce_first_stage, 1);
            noc_semaphore_inc(reduce_second_stage_receiver_semaphore_noc_addr, 1);
        }
    };
    global_reduce_receiver(cb_ex_partial2, cb_ex_external2, cb_ex2);
}
