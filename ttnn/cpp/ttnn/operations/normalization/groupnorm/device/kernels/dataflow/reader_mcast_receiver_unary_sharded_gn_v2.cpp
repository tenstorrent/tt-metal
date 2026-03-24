// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc_semaphore.h"

// split REDUCE across cores
void kernel_main() {
    constexpr uint32_t reduce_receiver_semaphore_id = get_compile_time_arg_val(0);
    constexpr uint32_t reduce_sender_semaphore_id = get_compile_time_arg_val(1);

    constexpr uint32_t num_batch_group = get_compile_time_arg_val(2);

    constexpr uint32_t per_core_N = get_compile_time_arg_val(3);
    const uint32_t per_core_N_bytes = get_compile_time_arg_val(4);
    const uint32_t per_core_N_bytes_with_stride = get_compile_time_arg_val(5);
    constexpr uint32_t per_core_M = get_compile_time_arg_val(6);
    constexpr uint32_t tile_height = get_compile_time_arg_val(7);

    const uint32_t mcast_sender_noc_x = get_arg_val<uint32_t>(0);
    const uint32_t mcast_sender_noc_y = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_ex_partial_id = tt::CBIndex::c_8;
    constexpr uint32_t cb_ex_id = tt::CBIndex::c_9;
    constexpr uint32_t cb_ex_global_id = tt::CBIndex::c_15;
    constexpr uint32_t cb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_repack_id = tt::CBIndex::c_11;
    constexpr uint32_t cb_repack_out_id = tt::CBIndex::c_12;
    constexpr uint32_t cb_out0_id = tt::CBIndex::c_16;

    experimental::Noc noc;
    experimental::Semaphore<> reduce_receiver_sem(reduce_receiver_semaphore_id);
    experimental::Semaphore<> reduce_sender_sem(reduce_sender_semaphore_id);
    experimental::CircularBuffer cb_ex_partial(cb_ex_partial_id);
    experimental::CircularBuffer cb_ex_global(cb_ex_global_id);
    experimental::CircularBuffer cb_in0(cb_in0_id);
    experimental::CircularBuffer cb_repack(cb_repack_id);
    experimental::CircularBuffer cb_repack_out(cb_repack_out_id);
    experimental::CircularBuffer cb_out0(cb_out0_id);

    const uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial_id);
    const DataFormat data_format = get_dataformat(cb_ex_partial_id);

#if defined(READER_REPACK) and defined(TILIZE_IN)
    uint32_t in0_l1_read_addr = cb_in0.get_read_ptr();
    uint64_t noc_addr_in0 = get_noc_addr(in0_l1_read_addr);
    for (uint32_t m = 0; m < per_core_M; ++m) {
        cb_repack.reserve_back(per_core_N);
        uint32_t l1_write_addr_repack = cb_repack.get_write_ptr();
        for (uint32_t i = 0; i < tile_height; ++i) {
            noc_async_read(noc_addr_in0, l1_write_addr_repack, per_core_N_bytes);
            noc_addr_in0 += per_core_N_bytes;
            l1_write_addr_repack += per_core_N_bytes_with_stride;
        }
        noc.async_read_barrier();
        cb_repack.push_back(per_core_N);
    }
#endif

    for (uint32_t i = 0; i < num_batch_group; ++i) {
        for (uint32_t j = 0; j < 2; ++j) {
            cb_ex_partial.wait_front(1);
            reduce_sender_sem.set(INVALID);
            cb_ex_global.reserve_back(1);
            reduce_receiver_sem.up(noc, mcast_sender_noc_x, mcast_sender_noc_y, 1);
            reduce_sender_sem.wait(VALID);
            cb_ex_global.push_back(1);
            cb_ex_partial.pop_front(1);
        }
    }

#if defined(READER_REPACK) and defined(UNTILIZE_OUT)
    uint32_t l1_write_addr_repack = cb_out0.get_write_ptr();
    for (uint32_t m = 0; m < per_core_M; ++m) {
        cb_repack_out.wait_front(per_core_N);
        uint32_t in0_l1_read_addr = cb_repack_out.get_read_ptr();
        uint64_t noc_addr_in0 = get_noc_addr(in0_l1_read_addr);
        for (uint32_t i = 0; i < tile_height; ++i) {
            noc_async_read(noc_addr_in0, l1_write_addr_repack, per_core_N_bytes);
            noc_addr_in0 += per_core_N_bytes_with_stride;
            l1_write_addr_repack += per_core_N_bytes;
        }
        noc.async_read_barrier();
        cb_repack_out.pop_front(per_core_N);
    }
#endif
}
