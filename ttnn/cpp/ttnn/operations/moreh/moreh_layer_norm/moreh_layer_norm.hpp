// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_layer_norm {
struct MorehLayerNorm {
    static std::vector<std::optional<Tensor>> invoke(
        const Tensor& input,
        const uint32_t normalized_dims,
        const float eps,
        const std::optional<const Tensor> gamma,
        const std::optional<const Tensor> beta,
        const std::optional<const Tensor> output,
        const std::optional<const Tensor> mean,
        const std::optional<const Tensor> rstd,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_layer_norm

namespace ttnn {
constexpr auto moreh_layer_norm =
    ttnn::register_operation<"ttnn::moreh_layer_norm", ttnn::operations::moreh::moreh_layer_norm::MorehLayerNorm>();
}
