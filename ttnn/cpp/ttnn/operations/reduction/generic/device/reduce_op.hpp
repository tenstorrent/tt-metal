// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

#include "ttnn/operation.hpp"
#include "common.hpp"

namespace ttnn::operations::reduction::generic::detail {

Tensor reduce(
    const Tensor& input_tensor,
    tt::tt_metal::ReduceOpMath reduce_math,
    tt::tt_metal::ReduceOpDim reduce_dim,
    float scaler = 1.0f,
    const tt::tt_metal::MemoryConfig& output_mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    const std::optional<tt::tt_metal::DataType>& output_dtype = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids = std::nullopt,
    bool negate = false,
    // When false (default), fp32 reduces run on the accurate SFPU path (full fp32); true selects the FPU.
    // Ignored for non-fp32 inputs.
    bool fast_and_approximate_mode = false,
    // Requested layout of the result; std::nullopt means "whatever the selected path emits":
    // ROW_MAJOR on the dense RM paths, TILE on the tilized ones.
    const std::optional<tt::tt_metal::Layout>& output_layout = std::nullopt);

}  // namespace ttnn::operations::reduction::generic::detail

namespace reduce_op_utils {

std::map<std::string, std::string> get_defines(
    tt::tt_metal::ReduceOpMath reduce_op, tt::tt_metal::ReduceOpDim reduce_dim);
}  // namespace reduce_op_utils
