// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/data_movement/bcast/bcast_types.hpp"
#include <tuple>

namespace ttnn::prim {

struct BcastParams {
    ttnn::BcastOpMath math_op;
    ttnn::BcastOpDim dim;
    tt::tt_metal::MemoryConfig output_mem_config;
    bool in_place = false;

    static constexpr auto attribute_names = std::forward_as_tuple("math_op", "dim", "output_mem_config", "in_place");
    auto attribute_values() const { return std::forward_as_tuple(math_op, dim, output_mem_config, in_place); }
};

struct BcastInputs {
    Tensor input_a;
    Tensor input_b;
    std::optional<Tensor> preallocated_output;

    static constexpr auto attribute_names = std::forward_as_tuple("input_a", "input_b", "preallocated_output");
    auto attribute_values() const { return std::forward_as_tuple(input_a, input_b, preallocated_output); }
};

}  // namespace ttnn::prim
