// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace ttnn::experimental::prim {

struct CreateQKVHeadsFromSeparateTensorsParams {
    const uint32_t num_q_heads;
    const uint32_t num_kv_heads;
    const uint32_t head_dim;
    const bool transpose_k_heads;
    const tt::tt_metal::MemoryConfig output_mem_config;
};

struct CreateQKVHeadsFromSeparateTensorsInputs {
    const Tensor& input_tensor;
    const Tensor& input_tensor_kv;
    const std::optional<std::array<Tensor, 3>>& optional_output_tensors;
};

using CreateQKVHeadsFromSeparateTensorsResult = std::tuple<Tensor, Tensor, Tensor>;
using CreateQKVHeadsFromSeparateTensorsResultSpec = std::tuple<TensorSpec, TensorSpec, TensorSpec>;

}  // namespace ttnn::experimental::prim
