// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

constexpr uint32_t reduce_call_count = get_compile_time_arg_val(0);
template <uint32_t I>
using ReduceCall = ttnn::kernel_lib::
    BoundReduceCallArgs<ttnn::kernel_lib::ReduceCallAtT<1, I>, dfb::in0, dfb::scaler, dfb::out, dfb::intermed1>;

void kernel_main() {
    constexpr uint32_t repetitions = get_arg(args::reduce_repetitions);
    using First = ReduceCall<0>;
    constexpr uint32_t startup_src_b =
        First::algorithm == compute_kernel_lib::ReduceAlgorithm::AccumulateViaAdd ? dfb::in0 : dfb::scaler;
    compute_kernel_hw_startup(dfb::in0, startup_src_b, dfb::out);

    for (uint32_t column = 0; column < get_arg(args::units_per_core); ++column) {
        for (uint32_t batch = 0; batch < repetitions; ++batch) {
            if (batch == 0) {
                compute_kernel_lib::reduce<ReduceCall<0>>();
            } else if constexpr (reduce_call_count > 1) {
                if (batch + 1 == repetitions) {
                    compute_kernel_lib::reduce<ReduceCall<reduce_call_count - 1>>();
                } else {
                    compute_kernel_lib::reduce<ReduceCall<1>>();
                }
            }
        }
    }
    DataflowBuffer(dfb::scaler).pop_front(get_arg(args::reduce_auxiliary_tiles));
}
