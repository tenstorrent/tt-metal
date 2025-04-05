// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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

namespace ttnn::operations::moreh::moreh_sgd {
struct MorehSgd {
    static std::vector<std::optional<Tensor>> invoke(
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

    static OptionalTensors create_async_optional_output_tensors(
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
};
}  // namespace ttnn::operations::moreh::moreh_sgd

namespace ttnn {
constexpr auto moreh_sgd =
    ttnn::register_operation_with_auto_launch_op<"ttnn::moreh_sgd", ttnn::operations::moreh::moreh_sgd::MorehSgd>();
}
