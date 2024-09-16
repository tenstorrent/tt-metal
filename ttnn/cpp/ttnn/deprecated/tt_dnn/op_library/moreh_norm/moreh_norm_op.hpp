// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "tt_metal/host_api.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

std::tuple<uint32_t, float, bool> get_floored_p_and_decimal_and_p_is_negative(float p);

operation::ProgramWithCallbacks moreh_norm_h_impl(const Tensor &input, float p, const Tensor &output, const ttnn::DeviceComputeKernelConfig compute_kernel_config);
operation::ProgramWithCallbacks moreh_norm_w_impl(const Tensor &input, float p, const Tensor &output, const ttnn::DeviceComputeKernelConfig compute_kernel_config);
operation::ProgramWithCallbacks moreh_norm_other_impl(const Tensor &input, float p, int64_t dim, const Tensor &output, const ttnn::DeviceComputeKernelConfig compute_kernel_config);

struct MorehNorm {
    float p;
    int64_t dim;
    bool keepdim;
    MemoryConfig memory_config;
    const ttnn::DeviceComputeKernelConfig compute_kernel_config;

    void validate_with_output_tensors(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<Tensor>> &output_tensors
        ) const;
    std::vector<tt::tt_metal::LegacyShape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<Tensor>>& output_tensors
    ) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple("p", "dim", "keepdim", "memory_config", "compute_kernel_config");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->p), std::cref(this->dim), std::cref(this->keepdim), std::cref(this->memory_config), std::cref(this->compute_kernel_config));
    }
};

Tensor moreh_norm(
    const Tensor &input,
    float p,
    std::optional<std::variant<int64_t, std::vector<int64_t>>> dim = std::nullopt,
    const bool keepdim = false,
    const std::optional<const Tensor> output = std::nullopt,
    const std::optional<MemoryConfig> &memory_config = std::nullopt,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

Tensor moreh_norm_impl(const Tensor &input, float p, int64_t dim,
    const bool keepdim = false,
    const std::optional<const Tensor> output = std::nullopt,
    const std::optional<MemoryConfig> &memory_config = std::nullopt,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace primary

}  // namespace operations

}  // namespace tt
