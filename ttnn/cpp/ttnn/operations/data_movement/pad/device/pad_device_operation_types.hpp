// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::operations::data_movement::pad {

struct operation_attributes_t {
    const ttnn::Shape output_logical_shape;
    const ttnn::Shape output_padded_shape;
    const ttnn::Shape input_tensor_start;
    const float pad_value;
    const tt::tt_metal::MemoryConfig output_mem_config;
    const bool use_multicore;
};

struct tensor_args_t {
    const Tensor& input;
    std::optional<Tensor> preallocated_output;
};

using tensor_return_value_t = Tensor;
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttnn::operations::data_movement::pad
