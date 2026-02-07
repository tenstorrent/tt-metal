// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

#include <optional>
#include <tuple>

namespace ttnn::prim {

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

    static constexpr auto attribute_names = std::forward_as_tuple("input_tensor", "opt_output");
    auto attribute_values() const { return std::forward_as_tuple(input_tensor, opt_output); }
};

}  // namespace ttnn::prim
