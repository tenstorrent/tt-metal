// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include <tuple>

namespace ttnn::experimental::prim {

struct ConcatenateHeadsParams {
    const CoreCoord compute_with_storage_grid_size;
    const tt::tt_metal::MemoryConfig output_mem_config;
};

struct ConcatenateHeadsInputs {
    const Tensor input;
    const std::optional<Tensor> preallocated_output;

    static constexpr auto attribute_names = std::forward_as_tuple("input", "preallocated_output");
    auto attribute_values() const { return std::forward_as_tuple(input, preallocated_output); }
};

}  // namespace ttnn::experimental::prim
