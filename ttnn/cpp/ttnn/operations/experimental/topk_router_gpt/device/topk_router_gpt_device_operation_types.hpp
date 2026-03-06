// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::experimental::topk_router_gpt {

struct operation_attributes_t {
    uint32_t k;              // Number of top experts per token (4)
    uint32_t num_experts;    // Total number of experts (128)
    bool untilize_output;    // If true, write output in ROW_MAJOR format
};

struct tensor_args_t {
    const Tensor& input_tensor;   // [B, hidden_dim] bf16
    const Tensor& weight_tensor;  // [hidden_dim, num_experts] bf16 in DRAM
    const Tensor& bias_tensor;    // [1, num_experts] bf16 in DRAM
};

// Two outputs: (indices_rm_u16, weights_rm)
// - indices_rm: [B, k_padded] uint16 ROW_MAJOR - expert indices
// - weights_rm: [B, k_padded] bf16 ROW_MAJOR - expert weights
using tensor_return_value_t = std::tuple<Tensor, Tensor>;
using spec_return_value_t = std::tuple<TensorSpec, TensorSpec>;

}  // namespace ttnn::operations::experimental::topk_router_gpt
