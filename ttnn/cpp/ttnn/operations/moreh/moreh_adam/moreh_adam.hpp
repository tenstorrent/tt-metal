// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn {

std::vector<std::optional<Tensor>> moreh_adam(
    const Tensor& param_in,
    const Tensor& grad,
    const Tensor& exp_avg_in,
    const Tensor& exp_avg_sq_in,
    std::optional<float> lr = 0.001f,
    std::optional<float> beta1 = 0.9f,
    std::optional<float> beta2 = 0.999f,
    std::optional<float> eps = 1e-8f,
    std::optional<float> weight_decay = 0.0f,
    std::optional<uint32_t> step = 0,
    std::optional<bool> amsgrad = false,
    const std::optional<const Tensor>& max_exp_avg_sq_in = std::nullopt,
    const std::optional<const Tensor>& param_out = std::nullopt,
    const std::optional<const Tensor>& exp_avg_out = std::nullopt,
    const std::optional<const Tensor>& exp_avg_sq_out = std::nullopt,
    const std::optional<const Tensor>& max_exp_avg_sq_out = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn
