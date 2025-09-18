// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
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
    constexpr uint32_t block_h_size_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t num_all_to_all_workers_first_stage = get_compile_time_arg_val(5);
    constexpr uint32_t num_tiles_per_worker = get_compile_time_arg_val(6);
    constexpr uint32_t num_tiles_per_worker_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t num_tiles_per_worker_last = get_compile_time_arg_val(8);
    constexpr uint32_t num_tiles_per_worker_last_bytes = get_compile_time_arg_val(9);
    constexpr bool row_major = (bool)get_compile_time_arg_val(10);
    constexpr uint32_t num_x = get_compile_time_arg_val(11);
    constexpr uint32_t num_y = get_compile_time_arg_val(12);
    constexpr bool use_two_stage_reduce = (bool)get_compile_time_arg_val(13);
    constexpr uint32_t num_blocks_first_stage = get_compile_time_arg_val(14);
    constexpr uint32_t num_blocks_second_stage = get_compile_time_arg_val(15);
    uint32_t reduce_second_stage_semaphore_addr = get_semaphore(get_compile_time_arg_val(16));

    const uint32_t mcast_dest_noc_start_x = get_arg_val<uint32_t>(0);
    const uint32_t mcast_dest_noc_start_y = get_arg_val<uint32_t>(1);
    const uint32_t mcast_dest_noc_end_x = get_arg_val<uint32_t>(2);
    const uint32_t mcast_dest_noc_end_y = get_arg_val<uint32_t>(3);
    const uint32_t start_x = get_arg_val<uint32_t>(4);
    const uint32_t start_y = get_arg_val<uint32_t>(5);

    tt_l1_ptr uint32_t* in0_remote_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(6));
    tt_l1_ptr uint32_t* in0_remote_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(6 + num_x));

    constexpr uint32_t cb_ex_partial = tt::CBIndex::c_8;
    constexpr uint32_t cb_ex = tt::CBIndex::c_9;
    constexpr uint32_t cb_ex_external = tt::CBIndex::c_10;
    constexpr uint32_t cb_ex_partial2 = tt::CBIndex::c_11;
    constexpr uint32_t cb_ex2 = tt::CBIndex::c_12;
    constexpr uint32_t cb_ex_external2 = tt::CBIndex::c_13;
    constexpr uint32_t cb_ex2pe = tt::CBIndex::c_20;
    constexpr uint32_t cb_ex_global = tt::CBIndex::c_15;  // E[x] global reduce

    const uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial2);
    const DataFormat data_format = get_dataformat(cb_ex_partial2);

    uint64_t remote_noc_addrs[num_blocks];

    uint32_t x = start_x, y = start_y;
    for (uint32_t i = 0; i < num_blocks; ++i) {
        remote_noc_addrs[i] = get_noc_addr(in0_remote_noc_x[x], in0_remote_noc_y[y], 0);
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

    const uint64_t multicast_data_noc = get_noc_multicast_addr(
        mcast_dest_noc_start_x, mcast_dest_noc_start_y, mcast_dest_noc_end_x, mcast_dest_noc_end_y, 0);

    const uint64_t reduce_sender_semaphore_noc_addr = multicast_data_noc | reduce_sender_semaphore_addr;

    volatile tt_l1_ptr uint32_t* reduce_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sender_semaphore_addr);
    volatile tt_l1_ptr uint32_t* reduce_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_receiver_semaphore_addr);
    volatile tt_l1_ptr uint32_t* reduce_second_stage_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_second_stage_semaphore_addr);

    const auto& global_reduce_sender = [&](
        const uint32_t cb_partial,
        const uint32_t cb_external,
        const uint32_t cb_ex,
        const uint32_t cb_ex_global,
        const uint32_t cb_reduce_first_stage) __attribute__((always_inline)) {
        // global reduce
        // wait for local data ready
        cb_wait_front(cb_partial, block_h);
        // inc semaphore of other cores, tell other all-to-all workers to start
        if constexpr (num_blocks > 1) {
            *reduce_sender_semaphore_addr_ptr = VALID;
            noc_semaphore_wait(reduce_receiver_semaphore_addr_ptr, num_blocks - 1);
            noc_semaphore_set(reduce_receiver_semaphore_addr_ptr, 0);
            noc_semaphore_set_multicast(reduce_sender_semaphore_addr, reduce_sender_semaphore_noc_addr, num_blocks - 1);
        }

        // read data from other cores - first stage reduce
        uint32_t l1_read_addr_ex_par = get_read_ptr(cb_partial);
        // read data from other cores - second stage reduce
        uint32_t l1_read_addr_ex = 0;
        uint32_t block_index_stride = 0;
        if constexpr (use_two_stage_reduce) {
            l1_read_addr_ex = get_read_ptr(cb_reduce_first_stage);
            if constexpr (row_major) {
                block_index_stride = num_x;
            } else {
                block_index_stride = num_y;
            }
        }
        // read from both stage
        for (uint32_t i = 0; i < num_tiles_per_worker; ++i) {
            // first stage
            cb_reserve_back(cb_external, num_blocks_first_stage);
            uint32_t l1_write_addr_external = get_write_ptr(cb_external);
            for (uint32_t block = 0; block < num_blocks_first_stage; ++block) {
                uint64_t noc_addr_ex_par = remote_noc_addrs[block] | l1_read_addr_ex_par;
                noc_async_read_one_packet(noc_addr_ex_par, l1_write_addr_external, single_tile_size_bytes);
                l1_write_addr_external += single_tile_size_bytes;
            }
            l1_read_addr_ex_par += single_tile_size_bytes;
            noc_async_read_barrier();
            cb_push_back(cb_external, num_blocks_first_stage);

            // sync with second-stage all-to-all workers
            if constexpr (use_two_stage_reduce) {
                if (i == 0) {
                    noc_semaphore_wait(reduce_second_stage_semaphore_addr_ptr, num_blocks_second_stage - 1);
                    noc_semaphore_set(reduce_second_stage_semaphore_addr_ptr, 0);
                }

                uint32_t curr_block_index = block_index_stride;
                cb_reserve_back(cb_external, num_blocks_second_stage - 1);
                for (uint32_t block = 0; block < num_blocks_second_stage - 1; ++block) {
                    uint64_t noc_addr_ex = remote_noc_addrs[curr_block_index] | l1_read_addr_ex;
                    noc_async_read_one_packet(noc_addr_ex, l1_write_addr_external, single_tile_size_bytes);
                    curr_block_index += block_index_stride;
                    l1_write_addr_external += single_tile_size_bytes;
                }
                l1_read_addr_ex += single_tile_size_bytes;
                noc_async_read_barrier();
                cb_push_back(cb_external, num_blocks_second_stage - 1);
            }
        }

        l1_read_addr_ex = get_read_ptr(cb_ex);
        uint32_t l1_write_addr_ex_global = get_write_ptr(cb_ex_global);
        cb_wait_front(cb_ex, num_tiles_per_worker);

        // sync with other all-to-all workers, on the same row
        if constexpr (num_all_to_all_workers_first_stage > 1) {
            noc_semaphore_wait(reduce_receiver_semaphore_addr_ptr, num_all_to_all_workers_first_stage - 1);
            noc_semaphore_set(reduce_receiver_semaphore_addr_ptr, 0);
        }

        // gather data to top row
        cb_reserve_back(cb_ex_global, block_h);
        for (uint32_t block = 0; block < num_all_to_all_workers_first_stage; ++block) {
            uint64_t noc_addr_ex = remote_noc_addrs[block] | l1_read_addr_ex;
            uint32_t num_tiles_bytes = block == num_all_to_all_workers_first_stage - 1 ? num_tiles_per_worker_last_bytes
                                                                                       : num_tiles_per_worker_bytes;
            if constexpr (num_tiles_per_worker_bytes <= NOC_MAX_BURST_SIZE) {
                noc_async_read_one_packet(noc_addr_ex, l1_write_addr_ex_global, num_tiles_bytes);
            } else {
                noc_async_read(noc_addr_ex, l1_write_addr_ex_global, num_tiles_bytes);
            }
            l1_write_addr_ex_global += num_tiles_bytes;
        }
        noc_async_read_barrier();

        // mcast
        uint32_t l1_read_addr_ex_global = get_read_ptr(cb_ex_global);
        cb_push_back(cb_ex_global, block_h);
        if constexpr (num_blocks > 1) {
            for (uint32_t block = 0; block < num_all_to_all_workers_first_stage; ++block) {
                *reduce_sender_semaphore_addr_ptr = block + 2;

                uint32_t num_tiles_bytes = block == num_all_to_all_workers_first_stage - 1
                                               ? num_tiles_per_worker_last_bytes
                                               : num_tiles_per_worker_bytes;

                noc_async_write_multicast(
                    l1_read_addr_ex_global,
                    multicast_data_noc | l1_read_addr_ex_global,
                    num_tiles_bytes,
                    num_blocks - 1,
                    true);
                noc_semaphore_set_multicast(
                    reduce_sender_semaphore_addr, reduce_sender_semaphore_noc_addr, num_blocks - 1);

                l1_read_addr_ex_global += num_tiles_bytes;
                noc_async_write_barrier();
            }
        }
    };
#ifndef RMSNORM
    global_reduce_sender(cb_ex_partial, cb_ex_external, cb_ex, cb_ex_global, cb_ex);
#endif
    global_reduce_sender(cb_ex_partial2, cb_ex_external2, cb_ex2pe, cb_ex_global, cb_ex2);
}
