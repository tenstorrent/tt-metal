// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/moreh_linear_backward_device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {

std::vector<std::optional<Tensor>> moreh_linear_backward(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& weight,
    const std::vector<bool>& are_required_outputs = {true, true, true},
    const std::optional<Tensor>& bias = std::nullopt,
    const std::optional<Tensor>& input_grad = std::nullopt,
    const std::optional<Tensor>& weight_grad = std::nullopt,
    const std::optional<Tensor>& bias_grad = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& input_grad_memory_config = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& weight_grad_memory_config = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& bias_grad_memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn
