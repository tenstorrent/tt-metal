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
    BinaryBackwardOpType op_type;
    tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::INVALID;
    // Per-slot mc (slot 0 = input_grad, slot 1 = other_grad). Caller resolves default-per-operand
    // so divergent input/other memory_config is served by the fused device op.
    std::array<tt::tt_metal::MemoryConfig, 2> output_memory_configs;
    // Per-slot grad emit mask; fused path now serves partial masks via writer-side
    // WRITE_{INPUT,OTHER}_GRAD gates (compute still runs both chains — see #63473).
    std::array<bool, 2> are_required_outputs{true, true};
};

struct BinaryBackwardInputs {
    const Tensor& grad_output;
    const Tensor& input;
    const Tensor& other;
    std::optional<Tensor> preallocated_input_grad;
    std::optional<Tensor> preallocated_other_grad;
};

}  // namespace ttnn::operations::binary_backward
