// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "hostdevcommon/common_values.hpp"

// split REDUCE across cores
void kernel_main() {
    constexpr uint32_t reduce_receiver_semaphore_id = get_compile_time_arg_val(0);
    constexpr uint32_t reduce_sender_semaphore_id = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t num_x = get_compile_time_arg_val(3);
    constexpr uint32_t num_y = get_compile_time_arg_val(4);
    constexpr bool use_two_stage_reduce = (bool)get_compile_time_arg_val(5);
    constexpr uint32_t num_blocks_first_stage = get_compile_time_arg_val(6);
    constexpr uint32_t num_blocks_second_stage = get_compile_time_arg_val(7);
    constexpr uint32_t reduce_second_stage_semaphore_id = get_compile_time_arg_val(8);
    constexpr uint32_t single_tile_size_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t cb_ex_partial2 = get_compile_time_arg_val(10);
    constexpr uint32_t cb_ex2 = get_compile_time_arg_val(11);
    constexpr uint32_t cb_ex_external2 = get_compile_time_arg_val(12);
    constexpr uint32_t post_reduce_sender_semaphore_id = get_compile_time_arg_val(13);
    constexpr uint32_t cb_stats_reduced = get_compile_time_arg_val(14);  // [E[x], E[x^2]] local to sender
    constexpr uint32_t cb_ex_global = get_compile_time_arg_val(15);      // [E[x], E[X^2]] global to all cores
    // num_mcast_dests = num_x * num_y, the cell count of the multicast bounding
    // box. For non-rectangular shard grids this differs from num_blocks (the
    // worker count). The NoC ack counter must be credited against the
    // rectangle size or noc_async_write_barrier() will wait forever.
    constexpr uint32_t num_mcast_dests = get_compile_time_arg_val(16);

    Noc noc_obj;
    DataflowBuffer cb_ex_partial2_obj(cb_ex_partial2);
    DataflowBuffer cb_ex2_obj(cb_ex2);
    DataflowBuffer cb_ex_external2_obj(cb_ex_external2);
    DataflowBuffer cb_stats_reduced_obj(cb_stats_reduced);
    DataflowBuffer cb_ex_global_obj(cb_ex_global);

    Semaphore<> post_reduce_sender_sem(post_reduce_sender_semaphore_id);
    Semaphore<> reduce_receiver_sem(reduce_receiver_semaphore_id);
    Semaphore<> reduce_sender_sem(reduce_sender_semaphore_id);
    Semaphore<> reduce_second_stage_sem(reduce_second_stage_semaphore_id);

    const uint32_t mcast_dest_noc_start_x = get_arg_val<uint32_t>(0);
    const uint32_t mcast_dest_noc_start_y = get_arg_val<uint32_t>(1);
    const uint32_t mcast_dest_noc_end_x = get_arg_val<uint32_t>(2);
    const uint32_t mcast_dest_noc_end_y = get_arg_val<uint32_t>(3);
    const uint32_t start_x = get_arg_val<uint32_t>(4);
    const uint32_t start_y = get_arg_val<uint32_t>(5);

    tt_l1_ptr uint32_t* in0_remote_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(6));
    tt_l1_ptr uint32_t* in0_remote_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(6 + num_x));

    const DataFormat data_format = get_dataformat(cb_ex_partial2);

    uint64_t remote_noc_addrs[num_blocks];

    uint32_t x = start_x, y = start_y;
    for (uint32_t i = 0; i < num_blocks; ++i) {
        remote_noc_addrs[i] = get_noc_addr(in0_remote_noc_x[x], in0_remote_noc_y[y], 0);
        ++x;
        if (x == num_x) {
            x = 0;
            ++y;
            if (y == num_y) {
                y = 0;
            }
        }
    }

    const auto& global_reduce_sender = [&](DataflowBuffer& cb_partial,
                                           DataflowBuffer& cb_external,
                                           DataflowBuffer& cb_reduce_first_stage) __attribute__((always_inline)) {
        // global reduce
        // wait for local data ready
        cb_partial.wait_front(1);  // TODO test for layernorm

        // inc semaphore of other cores, tell other all-to-all workers to start
        if constexpr (num_blocks > 1) {
            reduce_sender_sem.set(VALID);
            reduce_receiver_sem.wait(num_blocks - 1);
            reduce_receiver_sem.set(0);
            // num_dests counts the multicast bounding-box cells excluding self,
            // not the worker count.
            reduce_sender_sem.set_multicast(
                noc_obj,
                mcast_dest_noc_start_x,
                mcast_dest_noc_start_y,
                mcast_dest_noc_end_x,
                mcast_dest_noc_end_y,
                num_mcast_dests - 1);
        }

        // read data from other cores - first stage reduce
        uint32_t l1_read_addr_ex_par = cb_partial.get_read_ptr();
        // read from both stage
        // first stage
        cb_external.reserve_back(num_blocks_first_stage);
        uint32_t l1_write_addr_external = cb_external.get_write_ptr();
        for (uint32_t block = 0; block < num_blocks_first_stage; ++block) {
            uint64_t noc_addr_ex_par = remote_noc_addrs[block] | (l1_read_addr_ex_par);
            noc_async_read_one_packet(noc_addr_ex_par, l1_write_addr_external, single_tile_size_bytes);
            l1_write_addr_external += single_tile_size_bytes;
        }
        l1_read_addr_ex_par += single_tile_size_bytes;
        noc_obj.async_read_barrier();
        cb_external.push_back(num_blocks_first_stage);

        // sync with second-stage all-to-all workers
        if constexpr (use_two_stage_reduce) {
            uint32_t l1_read_addr_ex = cb_reduce_first_stage.get_read_ptr();
            uint32_t block_index_stride = num_x;
            reduce_second_stage_sem.wait(num_blocks_second_stage - 1);
            reduce_second_stage_sem.set(0);

            uint32_t curr_block_index = block_index_stride;
            for (uint32_t block = 0; block < num_blocks_second_stage - 1; ++block) {
                uint64_t noc_addr_ex = remote_noc_addrs[curr_block_index] | (l1_read_addr_ex);
                noc_async_read_one_packet(noc_addr_ex, l1_write_addr_external, single_tile_size_bytes);
                l1_write_addr_external += single_tile_size_bytes;
                curr_block_index += block_index_stride;
            }
            l1_read_addr_ex += single_tile_size_bytes;
            noc_obj.async_read_barrier();
            cb_external.push_back(
                (num_blocks_second_stage - 1));  // push back partials from all cores -> compute can start reducing now
        }
        cb_partial.pop_front(1);
    };
    global_reduce_sender(cb_ex_partial2_obj, cb_ex_external2_obj, cb_ex2_obj);

    const auto& global_semaphore_set = [&]() __attribute__((always_inline)) {
        post_reduce_sender_sem.set(VALID);

        // num_dests counts the multicast bounding-box cells (loopback includes
        // self), not the worker count.
        post_reduce_sender_sem.set_multicast<NocOptions::MCAST_INCL_SRC>(
            noc_obj,
            mcast_dest_noc_start_x,
            mcast_dest_noc_start_y,
            mcast_dest_noc_end_x,
            mcast_dest_noc_end_y,
            num_mcast_dests);
        noc_obj.async_write_barrier();
    };

    const auto& post_global_reduce_sender = [&](DataflowBuffer& cb_ex, DataflowBuffer& cb_ex_global_arg)
                                                __attribute__((always_inline)) {
                                                    uint32_t l1_read_addr_ex = cb_ex.get_read_ptr();
                                                    uint32_t l1_read_addr_ex_global = cb_ex_global_arg.get_read_ptr();
                                                    // num_dests counts the multicast bounding-box cells
                                                    // (loopback includes self), not the worker count.
                                                    noc_obj.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
                                                        CoreLocalMem<uint8_t>(l1_read_addr_ex),
                                                        MulticastEndpoint{},
                                                        single_tile_size_bytes,
                                                        num_mcast_dests,
                                                        {},
                                                        {.noc_x_start = mcast_dest_noc_start_x,
                                                         .noc_y_start = mcast_dest_noc_start_y,
                                                         .noc_x_end = mcast_dest_noc_end_x,
                                                         .noc_y_end = mcast_dest_noc_end_y,
                                                         .addr = l1_read_addr_ex_global},
                                                        false);
                                                    noc_obj.async_write_barrier();
                                                };
    cb_stats_reduced_obj.wait_front(1);
    cb_ex_global_obj.reserve_back(1);
    post_global_reduce_sender(cb_stats_reduced_obj, cb_ex_global_obj);
    cb_ex_global_obj.push_back(1);
    cb_stats_reduced_obj.pop_front(1);
    global_semaphore_set();
}
