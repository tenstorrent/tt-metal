// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/operations/moreh/moreh_softmax_backward/device/moreh_softmax_backward_device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {

Tensor moreh_softmax_backward(
    const Tensor& output_tensor,
    const Tensor& output_grad_tensor,
    uint32_t dim,
    const std::optional<Tensor>& input_grad_tensor = std::nullopt,
    const ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOp op =
        ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOp::SOFTMAX,
    const ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOpParallelizationStrategy strategy =
        ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOpParallelizationStrategy::NONE,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

Tensor moreh_softmin_backward(
    const Tensor& output_tensor,
    const Tensor& output_grad_tensor,
    uint32_t dim,
    const std::optional<Tensor>& input_grad_tensor = std::nullopt,
    const ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOp op =
        ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOp::SOFTMIN,
    const ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOpParallelizationStrategy strategy =
        ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOpParallelizationStrategy::NONE,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

Tensor moreh_logsoftmax_backward(
    const Tensor& output_tensor,
    const Tensor& output_grad_tensor,
    uint32_t dim,
    const std::optional<Tensor>& input_grad_tensor = std::nullopt,
    const ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOp op =
        ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOp::LOGSOFTMAX,
    const ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOpParallelizationStrategy strategy =
        ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOpParallelizationStrategy::NONE,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn
