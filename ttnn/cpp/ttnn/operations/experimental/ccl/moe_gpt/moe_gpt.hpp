// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include <tt-metalium/base_types.hpp>
#include <optional>
#include <vector>

namespace ttnn::operations::experimental::moe_gpt {

struct ExecuteMoEGPT {
    static std::vector<ttnn::Tensor> invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& expert_indices,
        const ttnn::Tensor& expert_scores,
        const ttnn::Tensor& expert_mapping,
        const ttnn::Tensor& w0_w1_tensor,
        const ttnn::Tensor& w2_tensor,
        uint32_t output_height_shard_dim = 4,
        uint32_t output_width_shard_dim = 3,
        uint32_t hidden_size = 2880,
        std::optional<uint32_t> cluster_axis = std::nullopt);
};

}  // namespace ttnn::operations::experimental::moe_gpt

namespace ttnn::experimental {
constexpr auto moe_gpt =
    ttnn::register_operation<"ttnn::experimental::moe_gpt", ttnn::operations::experimental::moe_gpt::ExecuteMoEGPT>();
}  // namespace ttnn::experimental
