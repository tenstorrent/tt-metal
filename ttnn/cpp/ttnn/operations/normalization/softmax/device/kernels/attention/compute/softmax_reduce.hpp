// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args.hpp"

constexpr uint32_t softmax_max_call_count = get_compile_time_arg_val(0);
constexpr uint32_t softmax_sum_offset =
    ttnn::kernel_lib::ReduceCallAtT<1, softmax_max_call_count - 1>::next_compile_time_args_offset();

template <
    PoolType Math,
    uint32_t Index,
    uint32_t Input,
    uint32_t Auxiliary,
    uint32_t Output,
    uint32_t Accumulator = Output>
using SoftmaxReduceCall = ttnn::kernel_lib::BoundReduceCallArgs<
    ttnn::kernel_lib::ReduceCallAtT<Math == PoolType::MAX ? 1 : softmax_sum_offset + 1, Index>,
    Input,
    Auxiliary,
    Output,
    Accumulator>;
