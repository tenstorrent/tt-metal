// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/base_types.hpp>
#include <optional>

namespace ttnn::operations::experimental::moe_gpt_fused {

struct operation_attributes_t {
    uint32_t num_experts{};
    uint32_t experts_per_device{4};
};

struct tensor_args_t {
    const Tensor& input_tensor;
    const Tensor& expert_indices;
    const Tensor& expert_scores;
    const Tensor& w0_w1_tensor;
    const Tensor& w2_tensor;
};

using tensor_return_value_t = std::vector<Tensor>;
using spec_return_value_t = std::vector<TensorSpec>;

}  // namespace ttnn::operations::experimental::moe_gpt_fused
