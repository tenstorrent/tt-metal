// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

#include "ttnn/operation.hpp"
#include "common.hpp"

namespace ttnn::operations::experimental::quasar::generic::detail {

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
    // When true, eligible mean/sum reduces consume ROW_MAJOR input directly via the dense
    // row-major fast path. When false (default), the op always tilizes and uses the classic
    // tile-reduce kernels. Default-off pending fixes to the dense RM path (perf regression +
    // multi-H-tile hang); see reduce_op.cpp for the eligibility constraints.
    bool use_row_major_support = false);

}  // namespace ttnn::operations::experimental::quasar::generic::detail

namespace reduce_op_utils_qsr {

std::map<std::string, std::string> get_defines(
    tt::tt_metal::ReduceOpMath reduce_op, tt::tt_metal::ReduceOpDim reduce_dim);
}  // namespace reduce_op_utils_qsr
