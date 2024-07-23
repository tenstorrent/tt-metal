/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

// The 'any' Tensor arg is only used to pass the device and resulting tensor dtype
struct MorehArange {
    float start;
    float end;
    float step;
    bool untilize_out;
    const std::optional<DataType> output_dtype;
    const CoreRange core_range;  // unused for now
    const MemoryConfig output_mem_config;

    void validate_with_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names =
        std::make_tuple("start", "end", "step", "untilize_out", "output_dtype", "output_mem_config");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->start),
            std::cref(this->end),
            std::cref(this->step),
            std::cref(this->untilize_out),
            std::cref(this->output_dtype),
            std::cref(this->output_mem_config));
    }
};

Tensor moreh_arange(float start, float end, float step,
    const Tensor& any,
    std::optional<Tensor> output_tensor = std::nullopt,
    bool untilize_out = false,
    std::optional<DataType> output_dtype=std::nullopt,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);


}  // namespace primary
}  // namespace operations
}  // namespace tt
