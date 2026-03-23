// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_softmax_backward.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {

Tensor moreh_softmax_backward(
    const Tensor& output_tensor,
    const Tensor& output_grad_tensor,
    const uint32_t dim,
    const std::optional<Tensor>& input_grad_tensor,
    const ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOp op,
    const ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOpParallelizationStrategy strategy,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::prim::moreh_softmax_backward(
        output_tensor, output_grad_tensor, dim, input_grad_tensor, op, strategy, memory_config, compute_kernel_config);
}

Tensor moreh_softmin_backward(
    const Tensor& output_tensor,
    const Tensor& output_grad_tensor,
    const uint32_t dim,
    const std::optional<Tensor>& input_grad_tensor,
    const ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOp op,
    const ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOpParallelizationStrategy strategy,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::prim::moreh_softmax_backward(
        output_tensor, output_grad_tensor, dim, input_grad_tensor, op, strategy, memory_config, compute_kernel_config);
}

Tensor moreh_logsoftmax_backward(
    const Tensor& output_tensor,
    const Tensor& output_grad_tensor,
    const uint32_t dim,
    const std::optional<Tensor>& input_grad_tensor,
    const ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOp op,
    const ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOpParallelizationStrategy strategy,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::prim::moreh_softmax_backward(
        output_tensor, output_grad_tensor, dim, input_grad_tensor, op, strategy, memory_config, compute_kernel_config);
}

}  // namespace ttnn
