// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_bmm_backward.hpp"

namespace ttnn {

std::vector<std::optional<Tensor>> moreh_bmm_backward(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& mat2,
    const std::vector<bool>& are_required_outputs,
    const std::optional<Tensor>& input_grad,
    const std::optional<Tensor>& mat2_grad,
    const std::optional<MemoryConfig>& input_grad_memory_config,
    const std::optional<MemoryConfig>& mat2_grad_memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    // TODO(nuked-op matmul): these were only consumed by the nuked moreh_matmul calls
    (void)mat2;
    (void)input_grad_memory_config;
    (void)mat2_grad_memory_config;
    (void)compute_kernel_config;
    std::vector<std::optional<Tensor>> outputs(2);
    if (are_required_outputs.at(0)) {
        TT_FATAL(input_grad.has_value(), "input_grad needs to have a value when input_requires_grad is True.");
        // TODO(nuked-op matmul): restore real call
        outputs[0] = output_grad;
    }
    if (are_required_outputs.at(1)) {
        TT_FATAL(mat2_grad.has_value(), "mat2_grad needs to have a value when mat2_requires_grad is True.");
        // TODO(nuked-op matmul): restore real call
        outputs[1] = input;
    }
    return outputs;
}

}  // namespace ttnn
