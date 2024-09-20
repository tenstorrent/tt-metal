/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

enum class MorehSoftmaxBackwardOpParallelizationStrategy {
    NONE,
    SMALL_W,
    SMALL_H,
    LARGE_W,
    LARGE_H,
    LARGE_C
};

enum class MorehSoftmaxBackwardOp {
    SOFTMAX,
    SOFTMIN,
    LOGSOFTMAX,
};

bool is_moreh_softmax_backward_w_small_available(const Tensor &tensor);
bool is_moreh_softmax_backward_h_small_available(const Tensor &tensor);

operation::ProgramWithCallbacks moreh_softmax_backward_w_small(
    const Tensor &output,
    const Tensor &output_grad,
    const Tensor &input_grad,
    const CoreRange core_range,
    const MorehSoftmaxBackwardOp op,
    const ttnn::DeviceComputeKernelConfig compute_kernel_config);
operation::ProgramWithCallbacks moreh_softmax_backward_w_large(
    const Tensor &output,
    const Tensor &output_grad,
    const Tensor &input_grad,
    const CoreRange core_range,
    const MorehSoftmaxBackwardOp op,
    const ttnn::DeviceComputeKernelConfig compute_kernel_config);
operation::ProgramWithCallbacks moreh_softmax_backward_h_small(
    const Tensor &output,
    const Tensor &output_grad,
    const Tensor &input_grad,
    const CoreRange core_range,
    const MorehSoftmaxBackwardOp op,
    const ttnn::DeviceComputeKernelConfig compute_kernel_config);
operation::ProgramWithCallbacks moreh_softmax_backward_h_large(
    const Tensor &output,
    const Tensor &output_grad,
    const Tensor &input_grad,
    const CoreRange core_range,
    const MorehSoftmaxBackwardOp op,
    const ttnn::DeviceComputeKernelConfig compute_kernel_config);
operation::ProgramWithCallbacks moreh_softmax_backward_c_large(
    const Tensor &output,
    const Tensor &output_grad,
    const Tensor &input_grad,
    uint32_t dim,
    const CoreRange core_range,
    const MorehSoftmaxBackwardOp op,
    const ttnn::DeviceComputeKernelConfig compute_kernel_config);

struct MorehSoftmaxBackward {
    const uint32_t dim;
    const CoreRange core_range;  // unused for now
    const MorehSoftmaxBackwardOp op;
    const MorehSoftmaxBackwardOpParallelizationStrategy strategy;
    const MemoryConfig output_mem_config;
    const ttnn::DeviceComputeKernelConfig compute_kernel_config;

    void validate_with_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    std::vector<tt::tt_metal::LegacyShape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    MorehSoftmaxBackwardOpParallelizationStrategy get_parallelization_strategy(
        const std::vector<Tensor> &input_tensors) const;
    static constexpr auto attribute_names = std::make_tuple("dim", "op", "strategy", "output_mem_config", "compute_kernel_config");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->dim), std::cref(this->op), std::cref(this->strategy), std::cref(this->output_mem_config), std::cref(this->compute_kernel_config));
    }
};

// const ref prevents
Tensor moreh_softmax_backward(
    const Tensor &output_tensor,
    const Tensor &output_grad_tensor,
    uint32_t dim,
    std::optional<Tensor> input_grad_tensor = std::nullopt,
    const MorehSoftmaxBackwardOpParallelizationStrategy strategy = MorehSoftmaxBackwardOpParallelizationStrategy::NONE,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

Tensor moreh_softmin_backward(
    const Tensor &output_tensor,
    const Tensor &output_grad_tensor,
    uint32_t dim,
    std::optional<Tensor> input_grad_tensor = std::nullopt,
    const MorehSoftmaxBackwardOpParallelizationStrategy strategy = MorehSoftmaxBackwardOpParallelizationStrategy::NONE,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

Tensor moreh_logsoftmax_backward(
    const Tensor &output_tensor,
    const Tensor &output_grad_tensor,
    uint32_t dim,
    std::optional<Tensor> input_grad_tensor = std::nullopt,
    const MorehSoftmaxBackwardOpParallelizationStrategy strategy = MorehSoftmaxBackwardOpParallelizationStrategy::NONE,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace primary
}  // namespace operations
}  // namespace tt
