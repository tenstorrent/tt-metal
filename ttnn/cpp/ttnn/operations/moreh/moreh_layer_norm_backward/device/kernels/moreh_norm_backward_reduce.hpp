// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args.hpp"
#include "experimental/kernel_args.h"

template <uint32_t Input, uint32_t Output, uint32_t Accumulator, uint32_t I>
using MorehGradCall = ttnn::kernel_lib::
    BoundReduceCallArgs<ttnn::kernel_lib::ReduceCallAtT<1, I>, Input, dfb::scaler, Output, Accumulator>;

template <uint32_t Input, uint32_t Output, uint32_t Accumulator>
ALWI void reduce_moreh_grad_block(uint32_t block, uint32_t num_blocks) {
    constexpr uint32_t call_count = get_compile_time_arg_val(0);
    if (block == 0) {
        compute_kernel_lib::reduce<MorehGradCall<Input, Output, Accumulator, 0>>();
    } else if constexpr (call_count > 1) {
        if (block + 1 == num_blocks) {
            compute_kernel_lib::reduce<MorehGradCall<Input, Output, Accumulator, call_count - 1>>();
        } else {
            compute_kernel_lib::reduce<MorehGradCall<Input, Output, Accumulator, 1>>();
        }
    }
}
