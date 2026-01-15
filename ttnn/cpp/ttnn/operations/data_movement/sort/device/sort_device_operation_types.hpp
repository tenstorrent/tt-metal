// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::sort {

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

using spec_return_value_t = std::vector<ttnn::TensorSpec>;
using tensor_return_value_t = std::vector<Tensor>;

}  // namespace ttnn::operations::data_movement::sort
