// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "hostdevcommon/common_values.hpp"

// split REDUCE across cores
void kernel_main() {
    constexpr uint32_t reduce_receiver_semaphore_id = get_compile_time_arg_val(0);
    constexpr uint32_t reduce_sender_semaphore_id = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);
    constexpr bool is_all_to_all_worker = get_compile_time_arg_val(3) == 1;
    constexpr uint32_t num_x = get_compile_time_arg_val(4);
    constexpr uint32_t num_y = get_compile_time_arg_val(5);
    constexpr bool use_two_stage_reduce = (bool)get_compile_time_arg_val(6);
    constexpr uint32_t num_blocks_first_stage = get_compile_time_arg_val(7);
    constexpr uint32_t num_blocks_second_stage = get_compile_time_arg_val(8);
    constexpr uint32_t reduce_second_stage_semaphore_id = get_compile_time_arg_val(9);
    constexpr uint32_t single_tile_size_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t cb_ex_partial2 = get_compile_time_arg_val(11);  // E[(x-E[x])^2] partial reduce
    constexpr uint32_t cb_ex2 = get_compile_time_arg_val(12);          // E[(x-E[x])^2] global reduce
    constexpr uint32_t cb_ex_external2 = get_compile_time_arg_val(13);
    constexpr uint32_t post_semaphore_id = get_compile_time_arg_val(14);
    constexpr uint32_t cb_ex_global = get_compile_time_arg_val(15);  // [E[x], E[X^2]] global to all cores

    Noc noc_obj;
    DataflowBuffer cb_ex_partial2_obj(cb_ex_partial2);
    DataflowBuffer cb_ex2_obj(cb_ex2);
    DataflowBuffer cb_ex_external2_obj(cb_ex_external2);
    DataflowBuffer cb_ex_global_obj(cb_ex_global);

    Semaphore<> post_reduce_sender_sem(post_semaphore_id);
    Semaphore<> reduce_second_stage_sem(reduce_second_stage_semaphore_id);
    Semaphore<> reduce_receiver_sem(reduce_receiver_semaphore_id);
    Semaphore<> reduce_sender_sem(reduce_sender_semaphore_id);

    post_reduce_sender_sem.set(INVALID);

    const uint32_t all_to_all_tile_offset_bytes = get_arg_val<uint32_t>(0);
    const bool is_second_stage_reader = get_arg_val<uint32_t>(1);
    const uint32_t start_x = get_arg_val<uint32_t>(2);
    const uint32_t start_y = get_arg_val<uint32_t>(3);
    volatile tt_l1_ptr uint32_t* in0_remote_noc_x = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(4));
    volatile tt_l1_ptr uint32_t* in0_remote_noc_y = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(4 + num_x));

    const DataFormat data_format = get_dataformat(cb_ex_partial2);  // data format

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

    // global reduce
    // wait for local data ready
    cb_ex_partial2_obj.wait_front(1);

    // inc mcast sender
    reduce_sender_sem.set(INVALID);
    reduce_receiver_sem.up(noc_obj, in0_remote_noc_x[0], in0_remote_noc_y[0], 1);
    reduce_sender_sem.wait(VALID);

    if constexpr (is_all_to_all_worker) {
        // read data from other cores - reduce first stage
        uint32_t l1_read_addr_ex_par = cb_ex_partial2_obj.get_read_ptr();
        l1_read_addr_ex_par += all_to_all_tile_offset_bytes;
        // read data from other cores - second stage reduce
        uint32_t l1_read_addr_ex = 0;
        [[maybe_unused]] uint32_t block_index_stride = 0;
        if constexpr (use_two_stage_reduce) {
            l1_read_addr_ex = cb_ex2_obj.get_read_ptr();
            block_index_stride = num_x;
        }
        cb_ex_external2_obj.reserve_back(num_blocks_first_stage);
        uint32_t l1_write_addr_external = cb_ex_external2_obj.get_write_ptr();

        for (uint32_t block = 0; block < num_blocks_first_stage; block++) {
            uint64_t noc_addr_ex_par =
                remote_noc_addrs_first_stage[block] | (l1_read_addr_ex_par);  // Updating read address for reading
                                                                              // SUm(X) and Sum(X2) per core
            noc_async_read_one_packet(noc_addr_ex_par, l1_write_addr_external, single_tile_size_bytes);
            l1_write_addr_external += single_tile_size_bytes;
        }
        l1_read_addr_ex_par += single_tile_size_bytes;
        noc_obj.async_read_barrier();
        cb_ex_external2_obj.push_back(num_blocks_first_stage);

        // read data from other cores - reduce first stage
        if constexpr (use_two_stage_reduce) {
            if (is_second_stage_reader) {  // gather data from a column of cores (if row major)
                reduce_second_stage_sem.wait(num_blocks_second_stage - 1);
                reduce_second_stage_sem.set(0);
                // read data from other cores - second stage reduce
                for (uint32_t block = 0; block < num_blocks_second_stage - 1; ++block) {
                    uint64_t noc_addr_ex = remote_noc_addrs_second_stage[block + 1] | l1_read_addr_ex;
                    noc_async_read_one_packet(noc_addr_ex, l1_write_addr_external, single_tile_size_bytes);
                    l1_write_addr_external += single_tile_size_bytes;
                }
                l1_read_addr_ex += single_tile_size_bytes;
                noc_obj.async_read_barrier();
                cb_ex_external2_obj.push_back(num_blocks_second_stage - 1);
            }
        }

        // sync with the gather worker
        cb_ex2_obj.wait_front(1);
        const uint64_t reduce_second_stage_receiver_semaphore_noc_addr =
            remote_noc_addrs_second_stage[0] | get_semaphore(reduce_second_stage_semaphore_id);
        noc_semaphore_inc(reduce_second_stage_receiver_semaphore_noc_addr, 1);
    }
    cb_ex_partial2_obj.pop_front(1);
    // Signal the compute kernel cb_ex_global ready
    cb_ex_global_obj.reserve_back(1);
    post_reduce_sender_sem.wait(VALID);
    cb_ex_global_obj.push_back(1);
    cb_ex2_obj.pop_front(1);
}
