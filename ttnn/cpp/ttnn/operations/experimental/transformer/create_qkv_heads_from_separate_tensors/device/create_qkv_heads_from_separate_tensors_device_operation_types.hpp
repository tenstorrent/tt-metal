// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace ttnn::operations::experimental::create_qkv_heads_from_separate_tensors {

struct operation_attributes_t {
    const uint32_t num_q_heads;
    const uint32_t num_kv_heads;
    const uint32_t head_dim;
    const bool transpose_k_heads;
    const tt::tt_metal::MemoryConfig output_mem_config;
};

struct tensor_args_t {
    const Tensor& input_tensor;
    const Tensor& input_tensor_kv;
    const std::optional<std::array<Tensor, 3>>& optional_output_tensors;
};

using spec_return_value_t = std::tuple<TensorSpec, TensorSpec, TensorSpec>;
using tensor_return_value_t = std::tuple<Tensor, Tensor, Tensor>;

}  // namespace ttnn::operations::experimental::create_qkv_heads_from_separate_tensors
