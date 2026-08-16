// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

void kernel_main() {
    using ReduceMcastArgs = dataflow_kernel_lib::McastArgs<0, 0>;
    constexpr ReduceMcastArgs reduce_mcast_args;
    constexpr uint32_t ct_base = ReduceMcastArgs::next_compile_time_args_offset();

    constexpr uint32_t block_h = get_compile_time_arg_val(ct_base + 3);
    constexpr bool rms_norm = get_compile_time_arg_val(ct_base + 15) == 1;

    constexpr uint32_t stats_tiles = rms_norm ? 1 : 2;

    DataflowBuffer dfb_ex_global_obj(tt::CBIndex::c_15);
    Noc noc;
    auto reduce_pipe = reduce_mcast_args.receiver(noc);

    dfb_ex_global_obj.reserve_back(stats_tiles * block_h);
    reduce_pipe.receive();
    dfb_ex_global_obj.push_back(stats_tiles * block_h);
}
