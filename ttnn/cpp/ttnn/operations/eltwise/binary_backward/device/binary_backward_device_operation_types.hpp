// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "binary_backward_op_types.hpp"

namespace ttnn::operations::binary_backward {

struct BinaryBackwardParams {
    const BinaryBackwardOpType op_type;
    const tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::INVALID;
    const tt::tt_metal::MemoryConfig output_memory_config;
    // Selects which grads the launched program emits. Initial contract for MUL_BW
    // requires both; partial masks are routed to the composite path in binary_backward.cpp.
    const std::array<bool, 2> are_required_outputs{true, true};
};

struct BinaryBackwardInputs {
    const Tensor& grad_output;
    const Tensor& input;
    const Tensor& other;
    std::optional<Tensor> preallocated_input_grad;
    std::optional<Tensor> preallocated_other_grad;
};

}  // namespace ttnn::operations::binary_backward
