// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/operations/moreh/moreh_softmax/device/moreh_softmax_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

Tensor moreh_softmax(
    const Tensor& input_tensor,
    uint32_t dim,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    operations::moreh::moreh_softmax::MorehSoftmaxOp op = operations::moreh::moreh_softmax::MorehSoftmaxOp::SOFTMAX,
    operations::moreh::moreh_softmax::MorehSoftmaxOpParallelizationStrategy strategy = operations::moreh::moreh_softmax::MorehSoftmaxOpParallelizationStrategy::NONE,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

Tensor moreh_softmin(
    const Tensor& input_tensor,
    uint32_t dim,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    operations::moreh::moreh_softmax::MorehSoftmaxOp op = operations::moreh::moreh_softmax::MorehSoftmaxOp::SOFTMIN,
    operations::moreh::moreh_softmax::MorehSoftmaxOpParallelizationStrategy strategy = operations::moreh::moreh_softmax::MorehSoftmaxOpParallelizationStrategy::NONE,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

Tensor moreh_logsoftmax(
    const Tensor& input_tensor,
    uint32_t dim,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    operations::moreh::moreh_softmax::MorehSoftmaxOp op = operations::moreh::moreh_softmax::MorehSoftmaxOp::LOGSOFTMAX,
    operations::moreh::moreh_softmax::MorehSoftmaxOpParallelizationStrategy strategy = operations::moreh::moreh_softmax::MorehSoftmaxOpParallelizationStrategy::NONE,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn
