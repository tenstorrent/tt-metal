// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    using Call =
        ttnn::kernel_lib::BoundReduceCallArgs<ttnn::kernel_lib::ReduceCallArgs<0>, dfb::input, dfb::scaler, dfb::out>;
    constexpr uint32_t startup_src_b =
        Call::algorithm == compute_kernel_lib::ReduceAlgorithm::AccumulateViaAdd ? dfb::input : dfb::scaler;
    compute_kernel_hw_startup(dfb::input, startup_src_b, dfb::out);

    // The reader streams each column contiguously. The plan handles its full
    // height, including the final partial tile, in one call.
    for (uint32_t column = 0; column < get_arg(args::units_per_core); ++column) {
        compute_kernel_lib::reduce<Call>();
    }
    DataflowBuffer(dfb::scaler).pop_front(Call::auxiliary_tile_count);
}
