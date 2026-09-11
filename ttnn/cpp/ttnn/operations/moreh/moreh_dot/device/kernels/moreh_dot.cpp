// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

constexpr uint32_t reduce_call_count = get_compile_time_arg_val(0);
template <uint32_t I>
using ReduceCall = ttnn::kernel_lib::
    BoundReduceCallArgs<ttnn::kernel_lib::ReduceCallAtT<1, I>, dfb::im0, dfb::scaler, dfb::out, dfb::im1>;

void kernel_main() {
    constexpr int onetile = 1;
    auto per_core_block_cnt = get_arg(args::per_core_block_cnt);
    compute_kernel_hw_startup(dfb::in0, dfb::in1, dfb::out);

    DataflowBuffer dfb_c0(dfb::in0);
    DataflowBuffer dfb_c1(dfb::in1);
    DataflowBuffer dfb_c2(dfb::scaler);
    DataflowBuffer dfb_c24(dfb::im0);

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        bool last_out = block == (per_core_block_cnt - 1);

        dfb_c0.wait_front(onetile);
        dfb_c1.wait_front(onetile);

        tile_regs_acquire();
        reconfig_data_format(dfb::in0, dfb::in1);
        pack_reconfig_data_format(dfb::im0);
        mul_init(dfb::in0, dfb::in1);
        mul_tiles(dfb::in0, dfb::in1, 0, 0, 0);
        tile_regs_commit();

        dfb_c0.pop_front(onetile);
        dfb_c1.pop_front(onetile);

        dfb_c24.reserve_back(onetile);

        tile_regs_wait();
        pack_tile(0, dfb::im0);
        tile_regs_release();

        dfb_c24.push_back(onetile);

        if (block == 0) {
            compute_kernel_lib::reduce<ReduceCall<0>>();
        } else {
            if constexpr (reduce_call_count > 1) {
                if (last_out) {
                    compute_kernel_lib::reduce<ReduceCall<reduce_call_count - 1>>();
                } else {
                    compute_kernel_lib::reduce<ReduceCall<1>>();
                }
            }
        }
    }
    dfb_c2.pop_front(get_arg(args::reduce_auxiliary_tiles));
}
