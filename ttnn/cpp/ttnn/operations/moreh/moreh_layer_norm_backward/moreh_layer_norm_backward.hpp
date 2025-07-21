// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_layer_norm_backward {
struct MorehLayerNormBackward {
    static std::vector<std::optional<Tensor>> invoke(
        const Tensor& output_grad,
        const Tensor& input,
        const Tensor& mean,
        const Tensor& rstd,
        uint32_t normalized_dims,
        const std::optional<const Tensor>& gamma,
        const std::optional<const Tensor>& input_grad,
        const std::optional<const Tensor>& gamma_grad,
        const std::optional<const Tensor>& beta_grad,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_layer_norm_backward

namespace ttnn {
constexpr auto moreh_layer_norm_backward = ttnn::register_operation<
    "ttnn::moreh_layer_norm_backward",
    ttnn::operations::moreh::moreh_layer_norm_backward::MorehLayerNormBackward>();
}  // namespace ttnn
