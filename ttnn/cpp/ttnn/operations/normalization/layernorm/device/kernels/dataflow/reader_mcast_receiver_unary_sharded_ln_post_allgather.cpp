// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

void kernel_main() {
    constexpr uint32_t block_h = get_compile_time_arg_val(2);
    constexpr bool rms_norm = get_compile_time_arg_val(14) == 1;

    constexpr uint32_t operation_ct_args_end = 16;
    constexpr uint32_t operation_rt_args_end = 0;
    constexpr dataflow_kernel_lib::McastArgs<operation_ct_args_end, operation_rt_args_end> final_statistics_mcast_args;

    constexpr uint32_t stats_tiles = rms_norm ? 1 : 2;

    DataflowBuffer dfb_ex_global_obj(tt::CBIndex::c_15);
    Noc noc;
    auto final_statistics_pipe = final_statistics_mcast_args.receiver(noc);

    dfb_ex_global_obj.reserve_back(stats_tiles * block_h);
    final_statistics_pipe.receive();
    dfb_ex_global_obj.push_back(stats_tiles * block_h);
}
