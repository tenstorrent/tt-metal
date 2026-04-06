
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {

Tensor moreh_nll_loss_backward(
    const Tensor& target_tensor,
    const Tensor& output_grad_tensor,
    bool reduction_mean,
    const std::optional<Tensor>& weight_tensor = std::nullopt,
    const std::optional<Tensor>& input_grad_tensor = std::nullopt,
    const std::optional<Tensor>& divisor_tensor = std::nullopt,
    int32_t ignore_index = -100,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn
