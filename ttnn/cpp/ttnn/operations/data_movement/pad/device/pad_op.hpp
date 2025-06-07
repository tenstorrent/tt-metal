// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::data_movement {

struct Pad {
    const ttnn::Shape output_logical_shape;
    const ttnn::Shape output_padded_shape;
    const ttnn::Shape input_tensor_start;
    const float pad_value;
    const tt::tt_metal::MemoryConfig output_mem_config;
    const bool use_multicore;

    void validate_with_output_tensors(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
    static constexpr auto attribute_names = std::forward_as_tuple(
        "output_logical_shape",
        "output_padded_shape",
        "input_tensor_start",
        "pad_value",
        "output_mem_config",
        "use_multicore");
    auto attribute_values() const {
        return std::forward_as_tuple(
            this->output_logical_shape,
            this->output_padded_shape,
            this->input_tensor_start,
            this->pad_value,
            this->output_mem_config,
            this->use_multicore);
    }
};

}  // namespace ttnn::operations::data_movement
