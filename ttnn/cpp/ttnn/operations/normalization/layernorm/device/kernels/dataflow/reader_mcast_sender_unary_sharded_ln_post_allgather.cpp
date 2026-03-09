// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc_semaphore.h"
#include "experimental/endpoints.h"

// split REDUCE across cores
void kernel_main() {
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t block_h = get_compile_time_arg_val(3);
    constexpr uint32_t block_h_size_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t num_tiles_per_worker = get_compile_time_arg_val(6);
    constexpr uint32_t num_tiles_per_worker_bytes = get_compile_time_arg_val(7);
    constexpr bool rms_norm = get_compile_time_arg_val(17) == 1;

    const uint32_t mcast_dest_noc_start_x = get_arg_val<uint32_t>(0);
    const uint32_t mcast_dest_noc_start_y = get_arg_val<uint32_t>(1);
    const uint32_t mcast_dest_noc_end_x = get_arg_val<uint32_t>(2);
    const uint32_t mcast_dest_noc_end_y = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_stats_reduced = tt::CBIndex::c_21;  // [E[x], E[x^2]] local to sender
    constexpr uint32_t cb_ex_global = tt::CBIndex::c_15;      // [E[x], E[X^2]] global to all cores

    experimental::Noc noc;
    experimental::Semaphore<> reduce_sender_sem(get_compile_time_arg_val(1));
    experimental::CircularBuffer cb_stats_reduced_obj(cb_stats_reduced);
    experimental::CircularBuffer cb_ex_global_obj(cb_ex_global);
    experimental::MulticastEndpoint mcast_ep;

    constexpr uint32_t stats_tiles = rms_norm ? 1 : 2;

    const auto& global_semaphore_set = [&]() __attribute__((always_inline)) {
        reduce_sender_sem.set(VALID);
        reduce_sender_sem.set_multicast<experimental::Noc::McastMode::INCLUDE_SRC>(
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
        [&](experimental::CircularBuffer& cb_ex_obj, experimental::CircularBuffer& cb_ex_global_obj_inner)
            __attribute__((always_inline)) {
                uint32_t l1_read_addr_ex_global = cb_ex_global_obj_inner.get_read_ptr();
                noc.async_write_multicast<experimental::Noc::McastMode::INCLUDE_SRC>(
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
