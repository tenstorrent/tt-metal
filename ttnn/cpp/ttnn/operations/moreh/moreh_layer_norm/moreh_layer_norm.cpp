// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_layer_norm.hpp"

#include "ttnn/operations/moreh/moreh_layer_norm/device/moreh_layer_norm_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_layer_norm {
std::vector<std::optional<Tensor>> MorehLayerNorm::invoke(
    const Tensor& input,
    const uint32_t normalized_dims,
    const float eps,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> beta,
    const std::optional<const Tensor> output,
    const std::optional<const Tensor> mean,
    const std::optional<const Tensor> rstd,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::prim::moreh_layer_norm(
        input,
        normalized_dims,
        eps,
        gamma,
        beta,
        output,
        mean,
        rstd,
        memory_config,
        compute_kernel_config);
}
}  // namespace ttnn::operations::moreh::moreh_layer_norm
