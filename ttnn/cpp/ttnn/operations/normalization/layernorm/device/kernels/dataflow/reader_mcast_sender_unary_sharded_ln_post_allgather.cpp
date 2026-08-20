// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args_spec.hpp"
#include "hostdevcommon/common_values.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"

// split REDUCE across cores
void kernel_main() {
    constexpr auto block_h = get_arg(args::block_h);
    constexpr auto num_tiles_per_worker_bytes = get_arg(args::num_tiles_per_worker_bytes);
#ifdef RMSNORM
    constexpr bool rms_norm = true;
#else
    constexpr bool rms_norm = false;
#endif

    Noc noc;
    // [E[x], E[x^2]] local to sender
    DataflowBuffer dfb_stats_reduced_obj(dfb::stats_reduced);
    // [E[x], E[X^2]] global to all cores
    DataflowBuffer dfb_ex_global_obj(dfb::ex_global);

    constexpr uint32_t stats_tiles = rms_norm ? 1 : 2;

    constexpr auto final_statistics = MCAST_SPEC_ARGS(final_statistics);
    auto final_statistics_pipe = final_statistics.sender(noc);

    dfb_stats_reduced_obj.wait_front(stats_tiles * block_h);
    dfb_ex_global_obj.reserve_back(stats_tiles * block_h);
    final_statistics_pipe.send(
        dfb_stats_reduced_obj.get_read_ptr(),
        dfb_ex_global_obj.get_write_ptr(),
        stats_tiles * num_tiles_per_worker_bytes);
    dfb_ex_global_obj.push_back(stats_tiles * block_h);
    dfb_stats_reduced_obj.pop_front(stats_tiles * block_h);
}
