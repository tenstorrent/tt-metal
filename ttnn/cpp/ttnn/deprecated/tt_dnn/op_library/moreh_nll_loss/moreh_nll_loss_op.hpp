/*
 * SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

operation::ProgramWithCallbacks moreh_nll_loss_step1_impl(
    const Tensor &target,
    const std::optional<const Tensor> weight,
    Tensor &output,
    const int32_t ignore_index,
    const bool reduction_mean,
    const uint32_t channel_size,
    const CoreRange core_range,
    const ttnn::DeviceComputeKernelConfig compute_kernel_config);

operation::ProgramWithCallbacks moreh_nll_loss_step2_impl(
    const Tensor &input,
    const Tensor &target,
    const std::optional<const Tensor> weight,
    const std::optional<const Tensor> divisor,
    Tensor &output,
    const int32_t ignore_index,
    const bool reduction_mean,
    const CoreRange core_range,
    const ttnn::DeviceComputeKernelConfig compute_kernel_config);

struct MorehNllLossStep1 {
    int32_t ignore_index;
    bool reduction_mean;
    const DataType output_dtype;
    const uint32_t channel_size;

    const MemoryConfig output_mem_config;
    const CoreRange core_range;  // unused for now
    const ttnn::DeviceComputeKernelConfig compute_kernel_config;

    void validate(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors) const;
    std::vector<tt::tt_metal::LegacyShape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        std::vector<Tensor> &output_tensors) const;
    static constexpr auto attribute_names = std::make_tuple(
        "ignore_index", "reduction_mean", "output_dtype", "channel_size", "output_mem_config", "compute_kernel_config");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->ignore_index),
            std::cref(this->reduction_mean),
            std::cref(this->output_dtype),
            std::cref(this->channel_size),
            std::cref(this->output_mem_config),
            std::cref(this->compute_kernel_config));
    }
};

struct MorehNllLossStep2 {
    int32_t ignore_index;
    bool reduction_mean;

    const MemoryConfig output_mem_config;
    const CoreRange core_range;  // unused for now
    const ttnn::DeviceComputeKernelConfig compute_kernel_config;

    void validate(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors) const;
    std::vector<tt::tt_metal::LegacyShape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        std::vector<Tensor> &output_tensors) const;
    static constexpr auto attribute_names =
        std::make_tuple("ignore_index", "reduction_mean", "output_mem_config", "compute_kernel_config");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->ignore_index),
            std::cref(this->reduction_mean),
            std::cref(this->output_mem_config),
            std::cref(this->compute_kernel_config));
    }
};

Tensor moreh_nll_loss_step1(
    const Tensor &target_tensor,
    const std::optional<const Tensor> weight_tensor,
    const int32_t ignore_index,
    const bool reduction_mean,
    const DataType output_dtype,
    const uint32_t channel_size,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

Tensor moreh_nll_loss_step2(
    const Tensor &input_tensor,
    const Tensor &target_tensor,
    const std::optional<const Tensor> weight_tensor,
    const std::optional<const Tensor> divisor_tensor,
    const int32_t ignore_index,
    const bool reduction_mean,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

Tensor moreh_nll_loss(
    const Tensor &input_tensor,
    const Tensor &target_tensor,
    const std::optional<const Tensor> weight_tensor,
    const std::optional<const Tensor> divisor_tensor,
    const std::optional<const Tensor> output_tensor,
    const int32_t ignore_index,
    const bool reduction_mean,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace primary
}  // namespace operations
}  // namespace tt
