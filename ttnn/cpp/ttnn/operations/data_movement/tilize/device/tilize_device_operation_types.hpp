// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement {

struct operation_attributes_t {
    const tt::tt_metal::MemoryConfig output_mem_config;
    const tt::tt_metal::DataType output_dtype;
    const bool use_multicore;
    const bool enough_space_width;
    const bool enough_space_height;
};

struct tensor_args_t {
    const std::vector<Tensor>& input_tensors;
    const std::vector<std::optional<const Tensor>>& optional_input_tensors;
    const std::vector<Tensor>& output_tensors;
};

using tensor_return_value_t = std::vector<Tensor>;
using spec_return_value_t = std::vector<ttnn::TensorSpec>;
}  // namespace ttnn::operations::data_movement
