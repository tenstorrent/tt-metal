// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::conv3d::detail {

tt::tt_metal::operation::ProgramWithCallbacks conv3d_factory(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const Tensor>& bias_tensor,
    const Conv3dConfig& config,
    const Tensor& output_tensor,
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::operations::experimental::conv3d::detail
