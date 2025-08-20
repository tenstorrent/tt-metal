// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/tt_metal.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "maxpool3d_device_operation.hpp"

namespace ttnn::operations::experimental::maxpool3d::detail {

tt::tt_metal::operation::ProgramWithCallbacks maxpool3d_factory(
    const Tensor& input_tensor,
    const MaxPool3dConfig& config,
    const Tensor& output_tensor,
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::operations::experimental::maxpool3d::detail
