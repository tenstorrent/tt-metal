// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

// split REDUCE across cores
void kernel_main() {
    constexpr uint32_t block_h = get_compile_time_arg_val(2);
    constexpr uint32_t num_tiles_per_worker_bytes = get_compile_time_arg_val(6);
    constexpr bool rms_norm = get_compile_time_arg_val(16) == 1;

    constexpr uint32_t operation_ct_args_end = 18;
    constexpr uint32_t operation_rt_args_end = 0;
    constexpr dataflow_kernel_lib::McastArgs<operation_ct_args_end, operation_rt_args_end> final_statistics_mcast_args;

    constexpr uint32_t dfb_stats_reduced = tt::CBIndex::c_21;  // [E[x], E[x^2]] local to sender
    constexpr uint32_t dfb_ex_global = tt::CBIndex::c_15;      // [E[x], E[X^2]] global to all cores

    Noc noc;
    DataflowBuffer dfb_stats_reduced_obj(dfb_stats_reduced);
    DataflowBuffer dfb_ex_global_obj(dfb_ex_global);
    auto final_statistics_pipe = final_statistics_mcast_args.sender(noc);

    constexpr uint32_t stats_tiles = rms_norm ? 1 : 2;

    dfb_stats_reduced_obj.wait_front(stats_tiles * block_h);
    dfb_ex_global_obj.reserve_back(stats_tiles * block_h);
    const uint32_t src_l1 = dfb_stats_reduced_obj.get_read_ptr();
    const uint32_t dst_l1 = dfb_ex_global_obj.get_write_ptr();
    const uint32_t size = stats_tiles * num_tiles_per_worker_bytes;
    final_statistics_pipe.send(src_l1, dst_l1, size);
    dfb_ex_global_obj.push_back(stats_tiles * block_h);
    dfb_stats_reduced_obj.pop_front(stats_tiles * block_h);
}
