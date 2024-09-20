/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <optional>

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

enum class MorehSoftmaxOpParallelizationStrategy {
    NONE,
    SMALL_W,
    SMALL_H,
    LARGE_W,
    LARGE_H,
    LARGE_C,
};

enum class MorehSoftmaxOp {
    SOFTMAX,
    SOFTMIN,
    LOGSOFTMAX,
};

bool is_moreh_softmax_w_small_available(const Tensor &tensor, const ttnn::DeviceComputeKernelConfig& compute_kernel_config);
bool is_moreh_softmax_h_small_available(const Tensor &tensor, const ttnn::DeviceComputeKernelConfig& compute_kernel_config);

operation::ProgramWithCallbacks moreh_softmax_w_small(
    const Tensor &input, const Tensor &output, const CoreRange core_range, const MorehSoftmaxOp op, const ttnn::DeviceComputeKernelConfig compute_kernel_config);
operation::ProgramWithCallbacks moreh_softmax_w_large(
    const Tensor &input, const Tensor &output, const CoreRange core_range, const MorehSoftmaxOp op, const ttnn::DeviceComputeKernelConfig compute_kernel_config);
operation::ProgramWithCallbacks moreh_softmax_h_small(
    const Tensor &input, const Tensor &output, const CoreRange core_range, const MorehSoftmaxOp op, const ttnn::DeviceComputeKernelConfig compute_kernel_config);
operation::ProgramWithCallbacks moreh_softmax_h_large(
    const Tensor &input, const Tensor &output, const CoreRange core_range, const MorehSoftmaxOp op, const ttnn::DeviceComputeKernelConfig compute_kernel_config);
operation::ProgramWithCallbacks moreh_softmax_c_large(
    const Tensor &input, const Tensor &output, uint32_t dim, const CoreRange core_range, const MorehSoftmaxOp op, const ttnn::DeviceComputeKernelConfig compute_kernel_config);

struct MorehSoftmax {
    const uint32_t dim;
    const CoreRange core_range;  // unused for now
    const MorehSoftmaxOp op;
    const MorehSoftmaxOpParallelizationStrategy strategy;
    const MemoryConfig output_mem_config;
    const ttnn::DeviceComputeKernelConfig compute_kernel_config;

    void validate_with_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    std::vector<tt::tt_metal::LegacyShape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    MorehSoftmaxOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
    static constexpr auto attribute_names = std::make_tuple("dim", "op", "strategy", "output_mem_config", "compute_kernel_config");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->dim), std::cref(this->op), std::cref(this->strategy), std::cref(this->output_mem_config), std::cref(this->compute_kernel_config));
    }
};

// const ref prevents
Tensor moreh_softmax(
    const Tensor &input_tensor,
    uint32_t dim,
    std::optional<Tensor> output_tensor = std::nullopt,
    const MorehSoftmaxOpParallelizationStrategy strategy = MorehSoftmaxOpParallelizationStrategy::NONE,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

Tensor moreh_softmin(
    const Tensor &input_tensor,
    uint32_t dim,
    std::optional<Tensor> output_tensor = std::nullopt,
    const MorehSoftmaxOpParallelizationStrategy strategy = MorehSoftmaxOpParallelizationStrategy::NONE,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

Tensor moreh_logsoftmax(
    const Tensor &input_tensor,
    uint32_t dim,
    std::optional<Tensor> output_tensor = std::nullopt,
    const MorehSoftmaxOpParallelizationStrategy strategy = MorehSoftmaxOpParallelizationStrategy::NONE,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace primary
}  // namespace operations
}  // namespace tt
