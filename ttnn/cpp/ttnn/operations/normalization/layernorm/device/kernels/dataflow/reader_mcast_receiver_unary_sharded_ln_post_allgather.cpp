// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

// split REDUCE across cores
void kernel_main() {
    constexpr uint32_t block_h = get_compile_time_arg_val(3);
    constexpr bool rms_norm = get_compile_time_arg_val(15) == 1;

    constexpr uint32_t dfb_ex_global = tt::CBIndex::c_15;

    constexpr uint32_t stats_tiles = rms_norm ? 1 : 2;

    Noc noc;
    DataflowBuffer dfb_ex_global_obj(dfb_ex_global);

    // This receiver does not acknowledge the sender, so its coordinates are semantically unused.
    // ReceiverPipe v9 still owns a fixed coordinate array; use this core as the inert entry.
    constexpr uint32_t reduce_sender_sem_id = get_compile_time_arg_val(1);
    const uint32_t unused_sender_coords[2] = {my_x[noc_index], my_y[noc_index]};
    dataflow_kernel_lib::ReceiverPipe<reduce_sender_sem_id, /*PRE_HANDSHAKE=*/false> reduce_pipe(
        noc, unused_sender_coords);

    dfb_ex_global_obj.reserve_back(stats_tiles * block_h);
    reduce_pipe.receive();
    dfb_ex_global_obj.push_back(stats_tiles * block_h);
}
