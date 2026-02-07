// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tuple>

namespace ttnn::prim {

struct ReshardParams {
    tt::tt_metal::MemoryConfig output_mem_config;

    static constexpr auto attribute_names = std::forward_as_tuple("output_mem_config");
    auto attribute_values() const { return std::forward_as_tuple(output_mem_config); }
};

struct ReshardInputs {
    Tensor input;
    std::optional<Tensor> preallocated_output;

    static constexpr auto attribute_names = std::forward_as_tuple("input", "preallocated_output");
    auto attribute_values() const { return std::forward_as_tuple(input, preallocated_output); }
};

}  // namespace ttnn::prim
