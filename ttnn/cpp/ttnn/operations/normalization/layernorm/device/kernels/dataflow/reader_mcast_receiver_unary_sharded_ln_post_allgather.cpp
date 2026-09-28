// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args_spec.hpp"
#include "hostdevcommon/common_values.hpp"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/dataflow_buffer.h"

// split REDUCE across cores
void kernel_main() {
    constexpr auto block_h = get_arg(args::block_h);
#ifdef RMSNORM
    constexpr bool rms_norm = true;
#else
    constexpr bool rms_norm = false;
#endif

    constexpr uint32_t stats_tiles = rms_norm ? 1 : 2;

    Noc noc;
    constexpr auto final_statistics = MCAST_SPEC_ARGS(final_statistics);
    auto final_statistics_pipe = final_statistics.receiver(noc);
    DataflowBuffer dfb_ex_global_obj(dfb::ex_global);

    dfb_ex_global_obj.reserve_back(stats_tiles * block_h);
    final_statistics_pipe.receive();
    dfb_ex_global_obj.push_back(stats_tiles * block_h);
}
