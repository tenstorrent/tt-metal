// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include <tuple>

namespace ttnn::prim {

struct CopyParams {
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype;
    bool backwards = false;

    static constexpr auto attribute_names = std::forward_as_tuple("output_mem_config", "output_dtype", "backwards");
    auto attribute_values() const { return std::forward_as_tuple(output_mem_config, output_dtype, backwards); }
};

struct CopyInputs {
    Tensor input;
    std::optional<Tensor> preallocated_output;

    static constexpr auto attribute_names = std::forward_as_tuple("input", "preallocated_output");
    auto attribute_values() const { return std::forward_as_tuple(input, preallocated_output); }
};

}  // namespace ttnn::prim
