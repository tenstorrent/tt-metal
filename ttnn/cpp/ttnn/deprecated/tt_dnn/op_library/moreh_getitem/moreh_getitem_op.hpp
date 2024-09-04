/*
 * SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>

#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

operation::ProgramWithCallbacks moreh_getitem_rm(
    const Tensor &input,
    const std::vector<Tensor> &index_tensors,
    const std::vector<uint32_t> &index_dims,
    const Tensor &output,
    const CoreRange core_range);

operation::ProgramWithCallbacks moreh_getitem_tilized(
    const Tensor &input,
    const std::vector<Tensor> &index_tensors,
    const std::vector<uint32_t> &index_dims,
    const Tensor &output,
    const CoreRange core_range);

struct MorehGetitem {
    const std::vector<uint32_t> index_dims;
    const CoreRange core_range;  // unused for now
    const MemoryConfig output_mem_config;

    void validate_with_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    static constexpr auto attribute_names = std::make_tuple("index_dims", "output_mem_config");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->index_dims), std::cref(this->output_mem_config));
    }
};

Tensor moreh_getitem(
    const Tensor &input_tensor,
    const std::vector<Tensor> &index_tensors,
    const std::vector<uint32_t> &index_dims,
    std::optional<Tensor> output_tensor = std::nullopt,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary
}  // namespace operations
}  // namespace tt
