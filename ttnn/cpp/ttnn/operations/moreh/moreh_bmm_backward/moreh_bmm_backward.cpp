// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_bmm_backward.hpp"

#include "ttnn/cpp/ttnn/operations/moreh/moreh_matmul/moreh_matmul.hpp"

namespace ttnn::operations::moreh::moreh_bmm_backward {
std::vector<std::optional<Tensor>> MorehBMMBackward::invoke(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& mat2,
    const std::vector<bool>& are_required_outputs,
    const std::optional<Tensor>& input_grad,
    const std::optional<Tensor>& mat2_grad,
    const std::optional<MemoryConfig>& input_grad_memory_config,
    const std::optional<MemoryConfig>& mat2_grad_memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    std::vector<std::optional<Tensor>> outputs(2);
    if (are_required_outputs.at(0)) {
        TT_FATAL(input_grad.has_value(), "input_grad needs to have a value when input_requires_grad is True.");
        outputs[0] = ttnn::moreh_matmul(
            output_grad,
            mat2,
            false,
            true,
            input_grad.value(),
            std::nullopt,
            input_grad_memory_config,
            compute_kernel_config);
    }
    if (are_required_outputs.at(1)) {
        TT_FATAL(mat2_grad.has_value(), "mat2_grad needs to have a value when mat2_requires_grad is True.");
        outputs[1] = ttnn::moreh_matmul(
            input,
            output_grad,
            true,
            false,
            mat2_grad.value(),
            std::nullopt,
            mat2_grad_memory_config,
            compute_kernel_config);
    }
    return outputs;
}
}  // namespace ttnn::operations::moreh::moreh_bmm_backward
