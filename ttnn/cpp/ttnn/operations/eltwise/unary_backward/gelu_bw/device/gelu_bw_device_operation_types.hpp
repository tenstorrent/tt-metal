// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::unary_backward::gelu_bw {

struct GeluBwParams {
    const tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::INVALID;
    const tt::tt_metal::MemoryConfig output_memory_config;
    const bool approximate = false;

    static constexpr auto attribute_names =
        std::forward_as_tuple("output_dtype", "output_memory_config", "approximate");
    auto attribute_values() const { return std::forward_as_tuple(output_dtype, output_memory_config, approximate); }
};

struct GeluBwInputs {
    const Tensor& grad_output;
    const Tensor& input;
    std::optional<Tensor> preallocated_input_grad;
};

}  // namespace ttnn::operations::unary_backward::gelu_bw
