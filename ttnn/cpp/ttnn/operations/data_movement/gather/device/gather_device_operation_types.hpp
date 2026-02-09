// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::gather {
struct GatherParams {
    const int8_t dim;
    const bool sparse_grad;
    const tt::tt_metal::MemoryConfig output_mem_config;
    const std::optional<CoreRangeSet> sub_core_grids;
};

struct GatherInputs {
    const Tensor& input_tensor;
    const Tensor& input_index_tensor;
    std::optional<Tensor> output_tensor;
};

using spec_return_value_t = ttnn::TensorSpec;
using tensor_return_value_t = Tensor;

}  // namespace ttnn::operations::data_movement::gather
