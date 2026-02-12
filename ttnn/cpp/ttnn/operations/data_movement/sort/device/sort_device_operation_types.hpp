// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include <tuple>

namespace ttnn::prim {

struct SortParams {
    const int8_t dim;
    const bool descending;
    const bool stable;
    const tt::tt_metal::MemoryConfig output_mem_config;

    static constexpr auto attribute_names = std::forward_as_tuple("dim", "descending", "stable", "output_mem_config");
    auto attribute_values() const { return std::forward_as_tuple(dim, descending, stable, output_mem_config); }
};

struct SortInputs {
    const Tensor& input_tensor;
    std::vector<std::optional<Tensor>> output_tensors;

    static constexpr auto attribute_names = std::forward_as_tuple("input_tensor", "output_tensors");
    auto attribute_values() const { return std::forward_as_tuple(input_tensor, output_tensors); }
};

}  // namespace ttnn::prim
