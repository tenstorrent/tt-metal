// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::experimental::topk_router_gpt {

struct operation_attributes_t {
    uint32_t k;            // Number of top experts per token (4)
    uint32_t num_experts;  // Total number of experts (128)
};

struct tensor_args_t {
    const Tensor& input_tensor;   // [B, hidden_dim] bf16
    const Tensor& weight_tensor;  // [hidden_dim, num_experts] bf16 in DRAM
    const Tensor& bias_tensor;    // [1, num_experts] bf16 in DRAM
    const Tensor& output_tensor;  // [B, num_experts] bf16 pre-allocated
};

// Output: single pre-allocated tensor
using tensor_return_value_t = Tensor;
using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::experimental::topk_router_gpt
