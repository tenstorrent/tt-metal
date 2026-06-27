// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

// split REDUCE across cores
void kernel_main() {
    constexpr uint32_t block_h = get_compile_time_arg_val(3);
    constexpr bool rms_norm = get_compile_time_arg_val(15) == 1;

    constexpr uint32_t cb_ex_global = tt::CBIndex::c_15;

    constexpr uint32_t stats_tiles = rms_norm ? 1 : 2;

    Noc noc;
    CircularBuffer cb_ex_global_obj(cb_ex_global);

    // mcast_pipe (C1 post_allgather): receiver side. The sender broadcasts the global reduce result
    // (INCLUDE_SRC) into a FRESH cb_ex_global slot + a VALID data-ready flag. NO pre-handshake (the
    // sender never gates on this core), so PRE_HANDSHAKE=false (CONSUMER_READY_SEM_ID omitted) and
    // receive() does NOT ack — the sender coords are unused (pass this core's own as a harmless dummy).
    // The ctor clears the flag (INVALID), folding in the old pre-loop reduce_sender_sem.set(INVALID).
    // data_ready = reduce_sender_sem (CTA 1). receive() waits VALID + clears for the next round; the
    // data lands in the reserved cb_ex_global slot via the sender's mcast (flag arrival => data arrival).
    constexpr uint32_t reduce_sender_sem_id = get_compile_time_arg_val(1);
    dataflow_kernel_lib::ReceiverPipe<reduce_sender_sem_id, /*PRE_HANDSHAKE=*/false> reduce_pipe(noc);

    cb_ex_global_obj.reserve_back(stats_tiles * block_h);
    reduce_pipe.receive(my_x[noc_index], my_y[noc_index]);
    cb_ex_global_obj.push_back(stats_tiles * block_h);
}
