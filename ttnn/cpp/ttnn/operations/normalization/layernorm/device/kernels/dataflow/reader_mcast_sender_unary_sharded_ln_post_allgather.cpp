// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
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

    constexpr uint32_t cb_stats_reduced = tt::CBIndex::c_21;  // [E[x], E[x^2]] local to sender
    constexpr uint32_t cb_ex_global = tt::CBIndex::c_15;      // [E[x], E[X^2]] global to all cores

    Noc noc;
    CircularBuffer cb_stats_reduced_obj(cb_stats_reduced);
    CircularBuffer cb_ex_global_obj(cb_ex_global);

    constexpr uint32_t stats_tiles = rms_norm ? 1 : 2;

    // mcast_pipe (C1 post_allgather): one-shot loopback broadcast of the global reduce result + a
    // VALID data-ready flag, INCLUDE_SRC. The sender reads its OWN reduced stats (cb_stats_reduced) and
    // writes them into cb_ex_global on ALL num_blocks cores INCLUDING itself (src != dst => true
    // loopback): NUM_ACTIVE_RECEIVER_CORES = num_blocks - 1 (recipients excluding self), the helper
    // adds +1 for the self-copy. data_ready = reduce_sender_sem (CTA 1, S->R level flag). NO
    // pre-handshake: each receiver mcasts into a FRESH reserve_back slot, so PRE_HANDSHAKE=false
    // (CONSUMER_READY_SEM_ID omitted). send() couples the data mcast + VALID flag (was the old split
    // global_reduce_sender + global_semaphore_set); the CB push/pop between them are local bookkeeping.
    constexpr uint32_t reduce_sender_sem_id = get_compile_time_arg_val(1);
    dataflow_kernel_lib::SenderPipe<noc_index, reduce_sender_sem_id, num_blocks - 1, /*PRE_HANDSHAKE=*/false>
        reduce_pipe(
            noc,
            dataflow_kernel_lib::McastRect<>{
                mcast_dest_noc_start_x, mcast_dest_noc_start_y, mcast_dest_noc_end_x, mcast_dest_noc_end_y});

    cb_stats_reduced_obj.wait_front(stats_tiles * block_h);
    cb_ex_global_obj.reserve_back(stats_tiles * block_h);
    reduce_pipe.send(
        cb_stats_reduced_obj.get_read_ptr(), cb_ex_global_obj.get_read_ptr(), stats_tiles * num_tiles_per_worker_bytes);
    cb_ex_global_obj.push_back(stats_tiles * block_h);
    cb_stats_reduced_obj.pop_front(stats_tiles * block_h);
}
