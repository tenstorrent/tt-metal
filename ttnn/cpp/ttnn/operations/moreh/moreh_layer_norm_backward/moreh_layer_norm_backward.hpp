// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_layer_norm_backward {
}  // namespace ttnn::operations::moreh::moreh_layer_norm_backward

namespace ttnn {

std::vector<std::optional<Tensor>> moreh_layer_norm_backward(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    uint32_t normalized_dims,
    const std::optional<const Tensor>& gamma = std::nullopt,
    const std::optional<const Tensor>& input_grad = std::nullopt,
    const std::optional<const Tensor>& gamma_grad = std::nullopt,
    const std::optional<const Tensor>& beta_grad = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn
