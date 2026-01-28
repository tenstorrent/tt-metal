// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct SortParams {
    const int8_t dim;
    const bool descending;
    const bool stable;
    const tt::tt_metal::MemoryConfig output_mem_config;
};

struct SortInputs {
    const Tensor& input_tensor;
    std::vector<std::optional<Tensor>> output_tensors;
};

}  // namespace ttnn::prim
