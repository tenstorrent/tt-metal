// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tuple>

namespace ttnn::experimental::prim {

struct NlpCreateQkvHeadsSegformerParams {
    tt::tt_metal::MemoryConfig output_mem_config;

    static constexpr auto attribute_names = std::forward_as_tuple("output_mem_config");
    auto attribute_values() const { return std::forward_as_tuple(output_mem_config); }
};

struct NlpCreateQkvHeadsSegformerInputs {
    Tensor input_tensor;
    std::vector<std::optional<Tensor>> optional_output_tensors;

    static constexpr auto attribute_names = std::forward_as_tuple("input_tensor", "optional_output_tensors");
    auto attribute_values() const { return std::forward_as_tuple(input_tensor, optional_output_tensors); }
};

using NlpCreateQkvHeadsSegformerResult = std::tuple<Tensor, Tensor, Tensor>;
using NlpCreateQkvHeadsSegformerResultSpec = std::tuple<TensorSpec, TensorSpec, TensorSpec>;

}  // namespace ttnn::experimental::prim
