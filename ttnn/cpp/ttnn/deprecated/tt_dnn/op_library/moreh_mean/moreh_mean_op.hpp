// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

struct MorehMean {
    int64_t dim;
    bool keepdim;
    const std::optional<uint32_t> divisor;
    MemoryConfig memory_config;
    const CoreRange core_range;  // unused for now
    const ttnn::DeviceComputeKernelConfig compute_kernel_config;

    void validate_with_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) const;

    static constexpr auto attribute_names =
        std::make_tuple("dim", "keepdim", "divisor", "memory_config", "compute_kernel_config");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->dim),
            std::cref(this->keepdim),
            std::cref(this->divisor),
            std::cref(this->memory_config),
            std::cref(this->compute_kernel_config));
    }
};

operation::ProgramWithCallbacks moreh_mean_nc(
    const Tensor &input,
    const Tensor &output,
    int64_t dim,
    const CoreRange core_range,
    const ttnn::DeviceComputeKernelConfig compute_kernel_config);
// revised from reduce_op
operation::ProgramWithCallbacks moreh_mean_w(
    const Tensor &input,
    const Tensor &output,
    const CoreRange core_range,
    const ttnn::DeviceComputeKernelConfig compute_kernel_config);
operation::ProgramWithCallbacks moreh_mean_h(
    const Tensor &input,
    const Tensor &output,
    const CoreRange core_range,
    const ttnn::DeviceComputeKernelConfig compute_kernel_config);

Tensor moreh_mean_(
    const Tensor &input,
    const int64_t &dim,
    const bool keepdim = false,
    const std::optional<uint32_t> divisor = std::nullopt,
    const std::optional<const Tensor> output = std::nullopt,
    const std::optional<MemoryConfig> memory_config = std::nullopt,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

std::optional<Tensor> moreh_mean(
    const Tensor &input,
    std::optional<std::variant<int64_t, std::vector<int64_t>>> dim = std::nullopt,
    const bool keepdim = false,
    const std::optional<uint32_t> divisor = std::nullopt,
    const std::optional<const Tensor> output = std::nullopt,
    const std::optional<MemoryConfig> memory_config = std::nullopt,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace primary

}  // namespace operations

}  // namespace tt
