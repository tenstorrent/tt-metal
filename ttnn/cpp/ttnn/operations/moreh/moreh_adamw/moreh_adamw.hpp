// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_adamw {}  // namespace ttnn::operations::moreh::moreh_adamw

namespace ttnn {

std::vector<std::optional<Tensor>> moreh_adamw(
    const Tensor& param_in,
    const Tensor& grad,
    const Tensor& exp_avg_in,
    const Tensor& exp_avg_sq_in,

    std::optional<float> lr,
    std::optional<float> beta1,
    std::optional<float> beta2,
    std::optional<float> eps,
    std::optional<float> weight_decay,
    std::optional<uint32_t> step,
    std::optional<bool> amsgrad,

    const std::optional<Tensor>& max_exp_avg_sq_in,
    const std::optional<Tensor>& param_out,
    const std::optional<Tensor>& exp_avg_out,
    const std::optional<Tensor>& exp_avg_sq_out,
    const std::optional<Tensor>& max_exp_avg_sq_out,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config);

}  // namespace ttnn
