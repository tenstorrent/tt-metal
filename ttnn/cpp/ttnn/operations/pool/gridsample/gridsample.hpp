// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn {
namespace operations {
namespace gridsample {

struct ExecuteGridSample {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& grid,
        const std::string& mode = std::string("bilinear"),
        const bool align_corners = false,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);
};
}  // namespace gridsample
}  // namespace operations
constexpr auto gridsample =
    ttnn::register_operation_with_auto_launch_op<"ttnn::gridsample", ttnn::operations::gridsample::ExecuteGridSample>();
}  // namespace ttnn
