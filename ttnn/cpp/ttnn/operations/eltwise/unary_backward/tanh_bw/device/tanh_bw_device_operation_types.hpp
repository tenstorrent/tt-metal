// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::unary_backward::tanh_bw {

struct TanhBwParams {
    const tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::INVALID;
    const tt::tt_metal::MemoryConfig output_memory_config;

    static constexpr auto attribute_names = std::forward_as_tuple("output_dtype", "output_memory_config");
    auto attribute_values() const { return std::forward_as_tuple(output_dtype, output_memory_config); }
};

struct TanhBwInputs {
    const Tensor& grad_output;
    const Tensor& input;
    std::optional<Tensor> preallocated_input_grad;
};

}  // namespace ttnn::operations::unary_backward::tanh_bw
