// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn {
namespace operations {
namespace experimental {
namespace conv3d {
struct Conv3dConfig;
}  // namespace conv3d
}  // namespace experimental
}  // namespace operations
}  // namespace ttnn

namespace ttnn::operations::experimental::conv3d::detail {

tt::tt_metal::operation::ProgramWithCallbacks conv3d_factory(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const Tensor>& bias_tensor,
    const Conv3dConfig& config,
    const Tensor& output_tensor,
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::operations::experimental::conv3d::detail
