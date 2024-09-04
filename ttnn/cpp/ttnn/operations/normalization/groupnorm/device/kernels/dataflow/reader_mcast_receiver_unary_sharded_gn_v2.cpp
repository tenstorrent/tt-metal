// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

// #include "debug/dprint.h"

// split REDUCE across cores
void kernel_main() {
    uint32_t reduce_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(0));
    uint32_t reduce_sender_semaphore_addr   = get_semaphore(get_compile_time_arg_val(1));

    constexpr uint32_t num_batch_group                  = get_compile_time_arg_val(2);

    constexpr uint32_t per_core_N                     = get_compile_time_arg_val(3);
    const uint32_t per_core_N_bytes                     = get_compile_time_arg_val(4);
    const uint32_t per_core_N_bytes_with_stride           = get_compile_time_arg_val(5);
    constexpr uint32_t per_core_M                      = get_compile_time_arg_val(6);
    constexpr uint32_t TILE_HEIGHT                      = get_compile_time_arg_val(7);


    const uint32_t mcast_sender_noc_x         = get_arg_val<uint32_t>(0);
    const uint32_t mcast_sender_noc_y         = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_ex_partial = tt::CB::dataflow0; // E[x] partial reduce
    constexpr uint32_t cb_ex = tt::CB::dataflow1; // E[x] partial reduce
    constexpr uint32_t cb_ex_global = tt::CB::dataflow7; // E[x] global reduce
    constexpr uint32_t cb_in0 = tt::CB::c_in0; // sharded cb
    constexpr uint32_t cb_repack = tt::CB::c_intermed2;
    constexpr uint32_t cb_repack_out = tt::CB::c_intermed7;
    constexpr uint32_t cb_out0 = tt::CB::c_out0;

    const uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial); // tile size
    const DataFormat data_format = get_dataformat(cb_ex_partial); // data format

    volatile tt_l1_ptr uint32_t* reduce_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_receiver_semaphore_addr);
    volatile tt_l1_ptr uint32_t* reduce_sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sender_semaphore_addr);

    const uint64_t reduce_receiver_semaphore_noc_addr = get_noc_addr(mcast_sender_noc_x, mcast_sender_noc_y, reduce_receiver_semaphore_addr);

    #if defined(READER_REPACK) and defined(TILIZE_IN)
    uint32_t in0_l1_read_addr = get_read_ptr(cb_in0);
    uint64_t noc_addr_in0 = get_noc_addr(in0_l1_read_addr);
    for (uint32_t m = 0; m < per_core_M; ++m) {
        cb_reserve_back(cb_repack, per_core_N);
        uint32_t l1_write_addr_repack = get_write_ptr(cb_repack);
        for (uint32_t i = 0; i < TILE_HEIGHT; ++i) {
            noc_async_read(noc_addr_in0, l1_write_addr_repack, per_core_N_bytes);
            noc_addr_in0 += per_core_N_bytes;
            l1_write_addr_repack += per_core_N_bytes_with_stride;
        }
        noc_async_read_barrier();
        cb_push_back(cb_repack, per_core_N);
    }
    #endif

    for (uint32_t i=0; i < num_batch_group; ++i) {
        for (uint32_t j=0; j < 2; ++j) {
            // wait for local data ready
            cb_wait_front(cb_ex_partial, 1);
            noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);
            cb_reserve_back(cb_ex_global, 1);
            noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);
            noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);
            cb_push_back(cb_ex_global, 1);
            cb_pop_front(cb_ex_partial, 1);
        }
    }

    #if defined(READER_REPACK) and defined(UNTILIZE_OUT)
    uint32_t l1_write_addr_repack = get_write_ptr(cb_out0);
    for (uint32_t m = 0; m < per_core_M; ++m) {
        cb_wait_front(cb_repack_out, per_core_N);
        uint32_t in0_l1_read_addr = get_read_ptr(cb_repack_out);
        uint64_t noc_addr_in0 = get_noc_addr(in0_l1_read_addr);
        for (uint32_t i = 0; i < TILE_HEIGHT; ++i) {
            noc_async_read(noc_addr_in0, l1_write_addr_repack, per_core_N_bytes);
            noc_addr_in0 += per_core_N_bytes_with_stride;
            l1_write_addr_repack += per_core_N_bytes;
        }
        noc_async_read_barrier();
        cb_pop_front(cb_repack_out, per_core_N);
    }
    #endif

}
