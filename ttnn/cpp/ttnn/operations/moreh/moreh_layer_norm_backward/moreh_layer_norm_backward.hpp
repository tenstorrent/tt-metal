// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <optional>
#include <vector>

#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt {
namespace tt_metal {
struct MemoryConfig;
}  // namespace tt_metal
}  // namespace tt

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

    // The parameters of this function must be identical to those of invoke.
    static OptionalTensors create_async_optional_output_tensors(
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
constexpr auto moreh_layer_norm_backward = ttnn::register_operation_with_auto_launch_op<
    "ttnn::moreh_layer_norm_backward",
    ttnn::operations::moreh::moreh_layer_norm_backward::MorehLayerNormBackward>();
}  // namespace ttnn
