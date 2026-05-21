// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "hostdevcommon/common_values.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"

// split REDUCE across cores
void kernel_main() {
    constexpr uint32_t num_blocks = get_arg(args::num_blocks);
    constexpr uint32_t block_h = get_arg(args::block_h);
    constexpr uint32_t block_h_size_bytes = get_arg(args::block_h_size_bytes);
    constexpr uint32_t num_tiles_per_worker = get_arg(args::num_tiles_per_worker);
    constexpr uint32_t num_tiles_per_worker_bytes = get_arg(args::num_tiles_per_worker_bytes);
    constexpr bool rms_norm = get_arg(args::rms_norm) == 1;

    const uint32_t mcast_dest_noc_start_x = get_arg(args::mcast_start_x);
    const uint32_t mcast_dest_noc_start_y = get_arg(args::mcast_start_y);
    const uint32_t mcast_dest_noc_end_x = get_arg(args::mcast_end_x);
    const uint32_t mcast_dest_noc_end_y = get_arg(args::mcast_end_y);

    constexpr uint32_t cb_stats_reduced = dfb::cb_stats_reduced;
    constexpr uint32_t cb_ex_global = dfb::cb_ex_global;

    Noc noc;
    Semaphore<> reduce_sender_sem(sem::reduce_sender);
    DataflowBuffer cb_stats_reduced_obj(cb_stats_reduced);
    DataflowBuffer cb_ex_global_obj(cb_ex_global);
    MulticastEndpoint mcast_ep;

    constexpr uint32_t stats_tiles = rms_norm ? 1 : 2;

    const auto& global_semaphore_set = [&]() __attribute__((always_inline)) {
        reduce_sender_sem.set(VALID);
        reduce_sender_sem.set_multicast<Noc::McastMode::INCLUDE_SRC>(
            noc,
            mcast_dest_noc_start_x,
            mcast_dest_noc_start_y,
            mcast_dest_noc_end_x,
            mcast_dest_noc_end_y,
            num_blocks,
            false);
        noc.async_write_barrier();
    };

    const auto& global_reduce_sender =
        [&](DataflowBuffer& cb_ex_obj, DataflowBuffer& cb_ex_global_obj_inner)
            __attribute__((always_inline)) {
                uint32_t l1_read_addr_ex_global = cb_ex_global_obj_inner.get_read_ptr();
                noc.async_write_multicast<Noc::McastMode::INCLUDE_SRC>(
                    cb_ex_obj,
                    mcast_ep,
                    stats_tiles * num_tiles_per_worker_bytes,
                    num_blocks,
                    {},
                    {.noc_x_start = mcast_dest_noc_start_x,
                     .noc_y_start = mcast_dest_noc_start_y,
                     .noc_x_end = mcast_dest_noc_end_x,
                     .noc_y_end = mcast_dest_noc_end_y,
                     .addr = l1_read_addr_ex_global},
                    false);
                noc.async_write_barrier();
            };

    cb_stats_reduced_obj.wait_front(stats_tiles * block_h);
    cb_ex_global_obj.reserve_back(stats_tiles * block_h);
    global_reduce_sender(cb_stats_reduced_obj, cb_ex_global_obj);
    cb_ex_global_obj.push_back(stats_tiles * block_h);
    cb_stats_reduced_obj.pop_front(stats_tiles * block_h);
    global_semaphore_set();
}
