// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_bmm_backward.hpp"
#include "ttnn/operations/moreh/moreh_matmul/device/moreh_matmul_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_bmm_backward {
std::vector<std::optional<Tensor>> MorehBmm::invoke(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& mat2,
    const std::vector<bool>& are_required_outputs,
    std::optional<const Tensor>& input_grad,
    std::optional<const Tensor>& mat2_grad,
    const MemoryConfig& input_grad_mem_config,
    const MemoryConfig& mat2_grad_mem_config,
    std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    std::vector<std::optional<Tensor>> outputs(2);
    std::cout << "came here 1\n";
    outputs.reserve(2);

    const bool input_requires_grad = are_required_outputs.at(0);
    const bool mat2_requires_grad = are_required_outputs.at(1);
    std::cout << "came here 2\n";

    if (input_requires_grad) {
        TT_FATAL(input_grad.has_value());
        std::cout << "came here 3\n";
        const auto& input_grad_tensor = input_grad.value();
        std::cout << "came here 4\n";
        outputs[0] = ttnn::prim::moreh_matmul(
            output_grad,
            mat2,
            false,
            true,
            input_grad_tensor,
            std::nullopt,
            input_grad_mem_config,
            compute_kernel_config);
        std::cout << "came here 5\n";
    }

    if (mat2_requires_grad) {
        TT_FATAL(mat2_grad.has_value());
        const auto& mat2_grad_tensor = mat2_grad.value();
        outputs[1] = ttnn::prim::moreh_matmul(
            input,
            output_grad,
            true,
            false,
            mat2_grad_tensor,
            std::nullopt,
            mat2_grad_mem_config,
            compute_kernel_config);
    }

    return outputs;
}
}  // namespace ttnn::operations::moreh::moreh_bmm_backward
