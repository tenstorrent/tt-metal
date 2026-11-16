// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"

#include "unary_backward_op_types.hpp"

namespace ttnn::operations::unary_backward {

struct UnaryBackwardParams {
    // Selects the compute kernel and discriminates program-cache entries between op types.
    const UnaryBackwardOpType op_type;
    const tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::INVALID;
    const tt::tt_metal::MemoryConfig output_memory_config;
};

struct UnaryBackwardInputs {
    const Tensor& grad_output;
    const Tensor& input;
    std::optional<Tensor> preallocated_input_grad;
};

}  // namespace ttnn::operations::unary_backward
