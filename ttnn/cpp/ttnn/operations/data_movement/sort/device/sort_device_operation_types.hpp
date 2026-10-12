// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
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
    // True when ttnn::sort padded the sort dim to a power of two (padding is ±inf, or 0 / 65535 for uint16).
    const bool sort_dim_padded;
};

struct SortInputs {
    const Tensor& input_tensor;
    std::vector<std::optional<Tensor>> output_tensors;
};

}  // namespace ttnn::prim
