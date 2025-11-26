// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::reduction {

struct operation_attributes_t {
    const tt::tt_metal::DataType output_dtype;
    const std::optional<int> dim;
    const bool keepdim;
    const std::optional<CoreRangeSet> sub_core_grids;
    const bool use_multicore;
    const tt::tt_metal::MemoryConfig output_mem_config;
};

struct tensor_args_t {
    const Tensor& input;
    std::optional<Tensor> optional_output_tensor;
};

using tensor_return_value_t = Tensor;
using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::reduction
