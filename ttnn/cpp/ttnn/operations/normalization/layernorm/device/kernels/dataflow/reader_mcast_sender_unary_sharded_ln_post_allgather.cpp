// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

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

    constexpr uint32_t dfb_stats_reduced = tt::CBIndex::c_21;  // [E[x], E[x^2]] local to sender
    constexpr uint32_t dfb_ex_global = tt::CBIndex::c_15;      // [E[x], E[X^2]] global to all cores

    Noc noc;
    DataflowBuffer dfb_stats_reduced_obj(dfb_stats_reduced);
    DataflowBuffer dfb_ex_global_obj(dfb_ex_global);

    constexpr uint32_t stats_tiles = rms_norm ? 1 : 2;

    // One-shot loopback broadcast of the global reduce result and its ready flag. The rectangle
    // covers all num_blocks cores including this sender, so SenderPipe derives the old INCLUDE_SRC
    // fan-out from the geometry. No pre-handshake is needed because every receiver reserves a fresh
    // cb_ex_global slot before waiting.
    constexpr uint32_t reduce_sender_sem_id = get_compile_time_arg_val(1);
    dataflow_kernel_lib::SenderPipe<noc_index, reduce_sender_sem_id, /*PRE_HANDSHAKE=*/false> reduce_pipe(
        noc,
        dataflow_kernel_lib::McastRect<>{
            mcast_dest_noc_start_x, mcast_dest_noc_start_y, mcast_dest_noc_end_x, mcast_dest_noc_end_y});

    dfb_stats_reduced_obj.wait_front(stats_tiles * block_h);
    dfb_ex_global_obj.reserve_back(stats_tiles * block_h);
    reduce_pipe.send(
        dfb_stats_reduced_obj.get_read_ptr(),
        dfb_ex_global_obj.get_read_ptr(),
        stats_tiles * num_tiles_per_worker_bytes);
    dfb_ex_global_obj.push_back(stats_tiles * block_h);
    dfb_stats_reduced_obj.pop_front(stats_tiles * block_h);
}
