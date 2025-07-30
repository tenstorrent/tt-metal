// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

#include <optional>

namespace ttnn::operations::reduction::accumulation {

enum class AccumulationOp : uint8_t { CUMSUM, CUMPROD };

struct operation_attributes_t {
    const int32_t dim;
    const DataType dtype;
    const MemoryConfig output_memory_config;
    const bool flip;
    const AccumulationOp op;
};

struct tensor_args_t {
    const Tensor& input_tensor;
    std::optional<Tensor> opt_output;
};

using spec_return_value_t = TensorSpec;
using tensor_return_value_t = Tensor;

}  // namespace ttnn::operations::reduction::accumulation
