// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/moreh_linear_backward_device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_linear_backward {
struct MorehLinearBackward {
    static std::tuple<bool, bool, bool> get_required_outputs(const std::vector<bool>& are_required_outputs);

    static std::vector<Tensor> create_async_output_tensors(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_inputs);

    static std::vector<std::optional<Tensor>> invoke(const Tensor& output_grad,
                                                     const Tensor& input,
                                                     const Tensor& weight,
                                                     const std::vector<bool>& are_required_outputs,
                                                     const std::optional<Tensor>& bias,
                                                     const std::optional<Tensor>& input_grad,
                                                     const std::optional<Tensor>& weight_grad,
                                                     const std::optional<Tensor>& bias_grad,
                                                     const std::optional<ttnn::MemoryConfig>& input_grad_memory_config,
                                                     const std::optional<ttnn::MemoryConfig>& weight_grad_memory_config,
                                                     const std::optional<ttnn::MemoryConfig>& bias_grad_memory_config,
                                                     const DeviceComputeKernelConfig compute_kernel_config);

    static std::vector<bool> create_async_return_flag(
        const Tensor& output_grad,
        const Tensor& input,
        const Tensor& weight,
        const std::vector<bool>& are_required_outputs,
        const std::optional<Tensor>& bias,
        const std::optional<Tensor>& input_grad,
        const std::optional<Tensor>& weight_grad,
        const std::optional<Tensor>& bias_grad,
        const std::optional<ttnn::MemoryConfig>& input_grad_memory_config,
        const std::optional<ttnn::MemoryConfig>& weight_grad_memory_config,
        const std::optional<ttnn::MemoryConfig>& bias_grad_memory_config,
        const DeviceComputeKernelConfig compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_linear_backward

namespace ttnn {
constexpr auto moreh_linear_backward =
    ttnn::register_operation_with_auto_launch_op<"ttnn::moreh_linear_backward",
                                                 ttnn::operations::moreh::moreh_linear_backward::MorehLinearBackward>();
}
