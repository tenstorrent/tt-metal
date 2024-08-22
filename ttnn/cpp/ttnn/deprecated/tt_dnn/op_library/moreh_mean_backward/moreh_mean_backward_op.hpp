/*
 * SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <vector>

#include "ttnn/deprecated/tt_dnn/op_library/compute_kernel_config.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

struct MorehMeanBackward {
    std::vector<int64_t> dims;
    bool keepdim;
    std::optional<Shape> input_grad_shape;
    MemoryConfig memory_config;
    const CoreRange core_range;  // unused for now
    const DeviceComputeKernelConfig compute_kernel_config;

    void validate_with_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names =
        std::make_tuple("dims", "keepdim", "input_grad_shape", "memory_config", "compute_kernel_config");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->dims),
            std::cref(this->keepdim),
            std::cref(this->input_grad_shape),
            std::cref(this->memory_config),
            std::cref(this->compute_kernel_config));
    }
};

operation::ProgramWithCallbacks moreh_mean_backward_impl(
    const Tensor &output_grad,
    const Tensor &input_grad,
    const std::vector<int64_t> &dims,
    const bool keepdim,
    const DeviceComputeKernelConfig &compute_kernel_config);

Tensor moreh_mean_backward_(
    const Tensor& output_grad,
    std::optional<std::variant<int64_t, std::vector<int64_t>>> dim,
    const bool keepdim,
    std::optional<Shape> input_grad_shape,
    const std::optional<const Tensor> input_grad,
    const std::optional<MemoryConfig> memory_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config);

Tensor moreh_mean_backward(
    const Tensor &output_grad,
    std::optional<std::variant<int64_t, std::vector<int64_t>>> dim = std::nullopt,
    const bool keepdim = false,
    std::optional<Shape> input_grad_shape = std::nullopt,
    const std::optional<const Tensor> input_grad = std::nullopt,
    const std::optional<MemoryConfig> memory_config = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace primary

}  // namespace operations

}  // namespace tt
