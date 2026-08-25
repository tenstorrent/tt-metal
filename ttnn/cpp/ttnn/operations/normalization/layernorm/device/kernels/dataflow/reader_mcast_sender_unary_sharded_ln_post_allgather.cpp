// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "hostdevcommon/common_values.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"

// split REDUCE across cores
void kernel_main() {
    constexpr auto num_blocks = get_arg(args::num_blocks);
    constexpr auto block_h = get_arg(args::block_h);
    constexpr auto num_tiles_per_worker_bytes = get_arg(args::num_tiles_per_worker_bytes);
#ifdef RMSNORM
    constexpr bool rms_norm = true;
#else
    constexpr bool rms_norm = false;
#endif

    const uint32_t mcast_dest_noc_start_x = get_arg(args::mcast_dest_noc_start_x);
    const uint32_t mcast_dest_noc_start_y = get_arg(args::mcast_dest_noc_start_y);
    const uint32_t mcast_dest_noc_end_x = get_arg(args::mcast_dest_noc_end_x);
    const uint32_t mcast_dest_noc_end_y = get_arg(args::mcast_dest_noc_end_y);

    Noc noc;
    Semaphore<> reduce_sender_sem(sem::reduce_sender);
    // [E[x], E[x^2]] local to sender
    DataflowBuffer dfb_stats_reduced_obj(dfb::stats_reduced);
    // [E[x], E[X^2]] global to all cores
    DataflowBuffer dfb_ex_global_obj(dfb::ex_global);
    MulticastEndpoint mcast_ep;

    constexpr uint32_t stats_tiles = rms_norm ? 1 : 2;

    const auto& global_semaphore_set = [&]() __attribute__((always_inline)) {
        reduce_sender_sem.set(VALID);
        reduce_sender_sem.set_multicast<NocOptions::MCAST_INCL_SRC>(
            noc,
            mcast_dest_noc_start_x,
            mcast_dest_noc_start_y,
            mcast_dest_noc_end_x,
            mcast_dest_noc_end_y,
            num_blocks,
            false);
        noc.async_write_barrier();
    };

    const auto& global_reduce_sender = [&](DataflowBuffer& dfb_ex_obj, DataflowBuffer& dfb_ex_global_obj_inner)
                                           __attribute__((always_inline)) {
                                               uint32_t l1_read_addr_ex_global = dfb_ex_global_obj_inner.get_read_ptr();
                                               noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
                                                   dfb_ex_obj,
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

    dfb_stats_reduced_obj.wait_front(stats_tiles * block_h);
    dfb_ex_global_obj.reserve_back(stats_tiles * block_h);
    global_reduce_sender(dfb_stats_reduced_obj, dfb_ex_global_obj);
    dfb_ex_global_obj.push_back(stats_tiles * block_h);
    dfb_stats_reduced_obj.pop_front(stats_tiles * block_h);
    global_semaphore_set();
}
