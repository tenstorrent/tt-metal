// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::conv3d::detail {

tt::tt_metal::operation::ProgramWithCallbacks conv3d_factory(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const Tensor>& bias_tensor,
    uint32_t output_channels,
    const std::array<uint32_t, 3>& kernel_size,
    const std::array<uint32_t, 3>& stride,
    const std::array<uint32_t, 3>& padding,
    const std::string& padding_mode,
    uint32_t groups,
    const std::optional<tt::tt_metal::DataType> dtype,
    const Conv3dConfig& config,
    const Tensor& output_tensor,
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::operations::experimental::conv3d::detail
