// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/tensor_utils.hpp>

#include "ttnn/operation.hpp"
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include <tracy/Tracy.hpp>
#include "tools/profiler/op_profiler.hpp"
#include <tt-metalium/reflection.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <type_traits>
#include <optional>

namespace tt::tt_metal::operation {

template <class Callable, class OutputType>
void launch_op(
    Callable&& op_func,
    const Tensors input_tensors,
    OutputType& output_tensors,
    const OptionalConstTensors optional_input_tensors,
    const OptionalTensors optional_output_tensors,
    bool enable_autoformat_device) {
    ZoneScopedN("LaunchOp");
    output_tensors = op_func(input_tensors, optional_input_tensors, optional_output_tensors);
}
}  // namespace tt::tt_metal::operation
