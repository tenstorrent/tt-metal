// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

constexpr uint32_t reduce_call_count = get_compile_time_arg_val(0);
template <uint32_t I>
using ReduceCall = ttnn::kernel_lib::
    BoundReduceCallArgs<ttnn::kernel_lib::ReduceCallAtT<1, I>, dfb::im0, dfb::scaler, dfb::out, dfb::im1>;

void kernel_main() {
    constexpr int onetile = 1;
    const auto per_core_block_cnt = get_arg(args::per_core_block_cnt);
    DataflowBuffer dfb_scaler(dfb::scaler);
    compute_kernel_hw_startup(dfb::in0, dfb::in1, dfb::out);

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        const bool last_out = block == (per_core_block_cnt - 1);

        ckl::mul<ckl::input(dfb::in0), ckl::input(dfb::in1), ckl::output(dfb::im0)>(
            ckl::IterationShape::tiles(onetile));

        if (block == 0) {
            ckl::reduce<ReduceCall<0>>();
        } else {
            if constexpr (reduce_call_count > 1) {
                if (last_out) {
                    ckl::reduce<ReduceCall<reduce_call_count - 1>>();
                } else {
                    ckl::reduce<ReduceCall<1>>();
                }
            }
        }
    }
    dfb_scaler.pop_front(get_arg(args::reduce_auxiliary_tiles));
}
