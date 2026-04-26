// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

enum class BinaryOpType : uint8_t { ADD = 0, MUL = 1 };

struct DitMinimalRmBinaryParams {
    const BinaryOpType op_type;
    const tt::tt_metal::DataType output_dtype;
    const tt::tt_metal::MemoryConfig output_memory_config;
};

struct DitMinimalRmBinaryInputs {
    const Tensor& input_a;
    const Tensor& input_b;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::experimental::prim
