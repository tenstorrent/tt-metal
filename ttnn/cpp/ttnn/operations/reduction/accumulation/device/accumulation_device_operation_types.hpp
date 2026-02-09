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

struct AccumulationParams {
    const int32_t dim;
    const DataType dtype;
    const MemoryConfig output_memory_config;
    const bool flip;
    const AccumulationOp op;
};

struct AccumulationInputs {
    const Tensor& input_tensor;
    std::optional<Tensor> opt_output;
};

}  // namespace ttnn::operations::reduction::accumulation
