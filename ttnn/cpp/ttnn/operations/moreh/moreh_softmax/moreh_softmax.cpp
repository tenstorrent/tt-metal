// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_softmax.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_softmax {

    Tensor MorehSoftmax::invoke(
        const Tensor &input_tensor,
        const uint32_t dim,
        const std::optional<Tensor> &output_tensor,
        const MorehSoftmaxOp op,
        const MorehSoftmaxOpParallelizationStrategy strategy,
        const std::optional<MemoryConfig> output_memory_config,
        const std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
            return ttnn::prim::moreh_softmax(
                input_tensor, dim, output_tensor, op, strategy, output_memory_config, compute_kernel_config);
        }

    Tensor MorehSoftmin::invoke(
        const Tensor &input_tensor,
        const uint32_t dim,
        const std::optional<Tensor> &output_tensor,
        const MorehSoftmaxOp op,
        const MorehSoftmaxOpParallelizationStrategy strategy,
        const std::optional<MemoryConfig> output_memory_config,
        const std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
            return ttnn::prim::moreh_softmax(
                input_tensor, dim, output_tensor, op, strategy, output_memory_config, compute_kernel_config);
        }

    Tensor MorehLogSoftmax::invoke(
        const Tensor &input_tensor,
        const uint32_t dim,
        const std::optional<Tensor> &output_tensor,
        const MorehSoftmaxOp op,
        const MorehSoftmaxOpParallelizationStrategy strategy,
        const std::optional<MemoryConfig> output_memory_config,
        const std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
            return ttnn::prim::moreh_softmax(
                input_tensor, dim, output_tensor, op, strategy, output_memory_config, compute_kernel_config);
        }
}
