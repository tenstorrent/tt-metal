// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_softmax_backward.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_softmax_backward {

    Tensor MorehSoftmaxBackward::invoke(
        const Tensor &output_tensor,
        const Tensor &output_grad_tensor,
        const uint32_t dim,
        const std::optional<Tensor> &input_grad_tensor,
        const MorehSoftmaxBackwardOp op,
        const MorehSoftmaxBackwardOpParallelizationStrategy strategy,
        const std::optional<MemoryConfig> output_memory_config,
        const std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
            return ttnn::prim::moreh_softmax_backward(
                output_tensor, output_grad_tensor, dim, input_grad_tensor, op, strategy, output_memory_config, compute_kernel_config);
        }

    Tensor MorehSoftminBackward::invoke(
        const Tensor &output_tensor,
        const Tensor &output_grad_tensor,
        const uint32_t dim,
        const std::optional<Tensor> &input_grad_tensor,
        const MorehSoftmaxBackwardOp op,
        const MorehSoftmaxBackwardOpParallelizationStrategy strategy,
        const std::optional<MemoryConfig> output_memory_config,
        const std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
            return ttnn::prim::moreh_softmax_backward(
                output_tensor, output_grad_tensor, dim, input_grad_tensor, op, strategy, output_memory_config, compute_kernel_config);
        }

    Tensor MorehLogSoftmaxBackward::invoke(
        const Tensor &output_tensor,
        const Tensor &output_grad_tensor,
        const uint32_t dim,
        const std::optional<Tensor> &input_grad_tensor,
        const MorehSoftmaxBackwardOp op,
        const MorehSoftmaxBackwardOpParallelizationStrategy strategy,
        const std::optional<MemoryConfig> output_memory_config,
        const std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
            return ttnn::prim::moreh_softmax_backward(
                output_tensor, output_grad_tensor, dim, input_grad_tensor, op, strategy, output_memory_config, compute_kernel_config);
        }
}
