// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tuple>

namespace ttnn::experimental::prim {

struct NlpConcatHeadsDecodeParams {
    uint32_t num_heads{};
    bool on_subcoregrids{};
    std::optional<CoreRangeSet> sub_core_grids;

    static constexpr auto attribute_names = std::forward_as_tuple("num_heads", "on_subcoregrids", "sub_core_grids");
    auto attribute_values() const { return std::forward_as_tuple(num_heads, on_subcoregrids, sub_core_grids); }
};

struct NlpConcatHeadsDecodeInputs {
    Tensor input;
    std::optional<Tensor> preallocated_output;

    static constexpr auto attribute_names = std::forward_as_tuple("input", "preallocated_output");
    auto attribute_values() const { return std::forward_as_tuple(input, preallocated_output); }
};

}  // namespace ttnn::experimental::prim
