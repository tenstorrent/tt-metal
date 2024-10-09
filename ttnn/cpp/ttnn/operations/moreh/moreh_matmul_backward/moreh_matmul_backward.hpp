// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
namespace ttnn::operations::moreh::moreh_matmul_backward {
struct MorehMatmulBackward {
    static std::vector<std::optional<Tensor>> invoke(
        const Tensor& output_grad,
        const Tensor& input,
        const Tensor& other,
        const std::vector<bool>& are_required_outputs,
        const std::optional<const Tensor>& input_grad,
        const std::optional<const Tensor>& other_grad,
        const std::optional<ttnn::MemoryConfig>& memory_config,
        const std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config);

    static std::vector<Tensor> create_async_output_tensors(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs);

    static std::vector<bool> create_async_return_flag(
        const Tensor& output_grad,
        const Tensor& input,
        const Tensor& other,
        const std::vector<bool>& are_required_outputs,
        const std::optional<const Tensor>& input_grad,
        const std::optional<const Tensor>& other_grad,
        const std::optional<ttnn::MemoryConfig>& memory_config,
        const std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_matmul_backward

namespace ttnn {
constexpr auto moreh_matmul_backward = ttnn::register_operation_with_auto_launch_op<
    "ttnn::moreh_matmul_backward",
    ttnn::operations::moreh::moreh_matmul_backward::MorehMatmulBackward>();
}
