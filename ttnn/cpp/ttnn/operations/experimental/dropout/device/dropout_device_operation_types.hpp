// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::dropout {

struct operation_attributes_t {
    const DataType output_dtype = DataType::INVALID;
    const MemoryConfig output_memory_config;
    uint32_t seed = 0;
    const float prob = 0.0f;
    const float scale = 1.0f;
};

struct tensor_args_t {
    const Tensor& input;
    std::optional<Tensor> preallocated_output;
};

using tensor_return_value_t = Tensor;

using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::experimental::dropout
