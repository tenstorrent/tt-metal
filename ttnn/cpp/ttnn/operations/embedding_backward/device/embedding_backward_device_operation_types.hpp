// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::embedding_backward::embedding_backward {

struct operation_attributes_t {
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::INVALID;
    uint32_t num_embeddings = 0;
};

struct tensor_args_t {
    Tensor index_tensor;
    Tensor grad_tensor;
    std::optional<Tensor> preallocated_output;
};

using spec_return_value_t = ttnn::TensorSpec;
using tensor_return_value_t = Tensor;

}  // namespace ttnn::operations::embedding_backward::embedding_backward
