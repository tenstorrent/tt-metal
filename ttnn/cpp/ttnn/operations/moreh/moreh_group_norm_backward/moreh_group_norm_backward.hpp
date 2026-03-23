// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {

std::vector<std::optional<Tensor>> moreh_group_norm_backward(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    uint32_t num_groups,
    const std::vector<bool>& are_required_outputs = {true, false, false},
    const std::optional<const Tensor>& gamma = std::nullopt,
    const std::optional<const Tensor>& input_grad = std::nullopt,
    const std::optional<const Tensor>& gamma_grad = std::nullopt,
    const std::optional<const Tensor>& beta_grad = std::nullopt,
    const std::optional<MemoryConfig>& input_grad_memory_config = std::nullopt,
    const std::optional<MemoryConfig>& gamma_grad_memory_config = std::nullopt,
    const std::optional<MemoryConfig>& beta_grad_memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn
