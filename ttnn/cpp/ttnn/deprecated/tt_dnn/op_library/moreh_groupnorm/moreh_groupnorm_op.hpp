// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

struct MorehGroupNorm {
    uint32_t num_groups;
    float eps;
    std::vector<bool> are_required_outputs;
    MemoryConfig output_mem_config;
    MemoryConfig mean_mem_config;
    MemoryConfig rstd_mem_config;

    void validate_with_output_tensors(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        const std::vector<std::optional<Tensor>> &output_tensors) const;

    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;

    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;

    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        std::vector<Tensor> &output_tensors) const;
};

operation::ProgramWithCallbacks moreh_groupnorm_impl(
    const Tensor &input,
    uint32_t num_groups,
    float eps,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> beta,
    Tensor &output,
    const std::optional<const Tensor> mean,
    const std::optional<const Tensor> rstd);

std::vector<std::optional<Tensor>> moreh_groupnorm(
    const Tensor &input,
    uint32_t num_groups,
    float eps,
    const std::optional<const Tensor> gamma = std::nullopt,
    const std::optional<const Tensor> beta = std::nullopt,
    const std::vector<bool> &are_required_outputs = std::vector<bool>{true, false, false},
    const std::optional<const Tensor> output = std::nullopt,
    const std::optional<const Tensor> mean = std::nullopt,
    const std::optional<const Tensor> rstd = std::nullopt,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    const MemoryConfig &mean_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    const MemoryConfig &rstd_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary

}  // namespace operations

}  // namespace tt
