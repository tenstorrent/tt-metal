// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include <tuple>

namespace ttnn::experimental::prim {

struct GeluBackwardParams {
    const tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::INVALID;
    const tt::tt_metal::MemoryConfig output_memory_config;
    const std::string approximate = "none";

    static constexpr auto attribute_names =
        std::forward_as_tuple("output_dtype", "output_memory_config", "approximate");
    auto attribute_values() const { return std::forward_as_tuple(output_dtype, output_memory_config, approximate); }
};

struct GeluBackwardInputs {
    const Tensor& grad_output;
    const Tensor& input;
    std::optional<Tensor> preallocated_input_grad;

    static constexpr auto attribute_names = std::forward_as_tuple("grad_output", "input", "preallocated_input_grad");
    auto attribute_values() const { return std::forward_as_tuple(grad_output, input, preallocated_input_grad); }
};

}  // namespace ttnn::experimental::prim
