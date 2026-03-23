// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {

std::vector<std::optional<Tensor>> moreh_sgd(
    const Tensor& param_in,
    const Tensor& grad,
    const std::optional<Tensor>& momentum_buffer_in,
    const std::optional<Tensor>& param_out,
    const std::optional<Tensor>& momentum_buffer_out,
    float lr,
    float momentum,
    float dampening,
    float weight_decay,
    bool nesterov,
    bool momentum_initialized,
    const std::optional<MemoryConfig>& param_out_memory_config,
    const std::optional<MemoryConfig>& momentum_buffer_out_memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
}
