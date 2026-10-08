// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    using MaxAuxiliary =
        ttnn::kernel_lib::BoundReduceAuxiliaryArgs<ttnn::kernel_lib::ReduceAuxiliaryArgs<0>, dfb::max_scaler>;
    using SumAuxiliary = ttnn::kernel_lib::BoundReduceAuxiliaryArgs<
        ttnn::kernel_lib::ReduceAuxiliaryArgs<MaxAuxiliary::next_compile_time_args_offset()>,
        dfb::sum_scaler>;
    dataflow_kernel_lib::prepare_reduce_auxiliary_tiles<MaxAuxiliary>();
    dataflow_kernel_lib::prepare_reduce_auxiliary_tiles<SumAuxiliary>();
}
