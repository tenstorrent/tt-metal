// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {

std::vector<std::optional<Tensor>> moreh_matmul_backward(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& other,
    const std::vector<bool>& are_required_outputs = {true, true},
    const std::optional<const Tensor>& input_grad = std::nullopt,
    const std::optional<const Tensor>& other_grad = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn
