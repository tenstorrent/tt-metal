// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/operations/moreh/moreh_softmax/device/moreh_softmax_device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_softmax {}  // namespace ttnn::operations::moreh::moreh_softmax

namespace ttnn {

using operations::moreh::moreh_softmax::MorehSoftmaxOp;
using operations::moreh::moreh_softmax::MorehSoftmaxOpParallelizationStrategy;

Tensor moreh_softmax(
    const Tensor& input_tensor,
    uint32_t dim,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    const MorehSoftmaxOp op = MorehSoftmaxOp::SOFTMAX,
    const MorehSoftmaxOpParallelizationStrategy strategy = MorehSoftmaxOpParallelizationStrategy::NONE,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

Tensor moreh_softmin(
    const Tensor& input_tensor,
    uint32_t dim,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    const MorehSoftmaxOp op = MorehSoftmaxOp::SOFTMIN,
    const MorehSoftmaxOpParallelizationStrategy strategy = MorehSoftmaxOpParallelizationStrategy::NONE,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

Tensor moreh_logsoftmax(
    const Tensor& input_tensor,
    uint32_t dim,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    const MorehSoftmaxOp op = MorehSoftmaxOp::LOGSOFTMAX,
    const MorehSoftmaxOpParallelizationStrategy strategy = MorehSoftmaxOpParallelizationStrategy::NONE,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn
