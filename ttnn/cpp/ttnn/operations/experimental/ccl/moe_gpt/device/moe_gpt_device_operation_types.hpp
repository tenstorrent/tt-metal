// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/base_types.hpp>
#include <optional>
#include <vector>

namespace ttnn::operations::experimental::moe_gpt {

struct operation_attributes_t {
    uint32_t output_height_shard_dim{4};
    uint32_t output_width_shard_dim{3};
    uint32_t hidden_size{2880};
    std::optional<uint32_t> cluster_axis;
};

struct tensor_args_t {
    const Tensor& input_tensor;
    const Tensor& expert_indices;
    const Tensor& expert_scores;
    const Tensor& expert_mapping;
    const Tensor& w0_w1_tensor;
    const Tensor& w2_tensor;
};

using tensor_return_value_t = std::vector<Tensor>;

using spec_return_value_t = std::vector<TensorSpec>;

}  // namespace ttnn::operations::experimental::moe_gpt
