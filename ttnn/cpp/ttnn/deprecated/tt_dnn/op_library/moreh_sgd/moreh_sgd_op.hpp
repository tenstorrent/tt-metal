/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <optional>

#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

operation::ProgramWithCallbacks moreh_sgd_(
    const Tensor& param_in,
    const Tensor& grad,
    std::optional<const Tensor> momentum_buffer_in,
    const Tensor& param_out,
    std::optional<const Tensor> momentum_buffer_out,
    float lr,
    float momentum,
    float dampening,
    float weight_decay,
    bool nesterov,
    bool momentum_initialized,
    const CoreRange core_range,
    const ttnn::DeviceComputeKernelConfig compute_kernel_config);

struct MorehSGD {
    float lr;
    float momentum;
    float dampening;
    float weight_decay;
    bool nesterov;
    bool momentum_initialized;
    const CoreRange core_range;  // unused for now
    MemoryConfig param_out_mem_config;
    MemoryConfig momentum_buffer_out_mem_config;
    const ttnn::DeviceComputeKernelConfig compute_kernel_config;

    void validate_with_output_tensors(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        const std::vector<std::optional<Tensor>> &output_tensors) const;
    std::vector<ttnn::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        std::vector<Tensor> &output_tensors) const;
    static constexpr auto attribute_names =
        std::make_tuple("lr", "momentum", "dampening", "weight_decay", "nesterov", "momentum_initialized", "param_out_mem_config", "momentum_buffer_out_mem_config", "compute_kernel_config");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->lr),
            std::cref(this->momentum),
            std::cref(this->dampening),
            std::cref(this->weight_decay),
            std::cref(this->nesterov),
            std::cref(this->momentum_initialized),
            std::cref(this->param_out_mem_config),
            std::cref(this->momentum_buffer_out_mem_config),
            std::cref(this->compute_kernel_config));
    }
};

std::vector<std::optional<Tensor>> moreh_sgd(
    const Tensor &param_in,
    const Tensor &grad,
    std::optional<const Tensor> momentum_buffer_in,
    std::optional<const Tensor> param_out,
    std::optional<const Tensor> momentum_buffer_out,
    float lr,
    float momentum,
    float dampening,
    float weight_decay,
    bool nesterov,
    bool momentum_initialized,
    const MemoryConfig &param_out_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    const MemoryConfig &momentum_buffer_out_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace primary
}  // namespace operations
}  // namespace tt
