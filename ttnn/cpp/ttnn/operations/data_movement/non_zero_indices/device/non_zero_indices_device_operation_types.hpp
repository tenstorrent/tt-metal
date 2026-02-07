// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tuple>

namespace ttnn::prim {

struct NonzeroParams {
    tt::tt_metal::MemoryConfig output_memory_config;

    static constexpr auto attribute_names = std::forward_as_tuple("output_memory_config");
    auto attribute_values() const { return std::forward_as_tuple(output_memory_config); }
};

struct NonzeroInputs {
    Tensor input;

    static constexpr auto attribute_names = std::forward_as_tuple("input");
    auto attribute_values() const { return std::forward_as_tuple(input); }
};

using NonzeroResult = std::tuple<Tensor, Tensor>;
using NonzeroResultSpec = std::tuple<TensorSpec, TensorSpec>;

}  // namespace ttnn::prim
