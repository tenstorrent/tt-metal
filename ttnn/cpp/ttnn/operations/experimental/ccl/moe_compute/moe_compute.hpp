// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteMoECompute {
    static std::vector<ttnn::Tensor> invoke(
        const ttnn::Tensor& tilize_input_tensor,
        const ttnn::Tensor& tilize_expert_indices_tensor,
        const ttnn::Tensor& tilize_expert_scores_tensor,
        const ttnn::Tensor& tilize_expert_mapping_tensor,
        const ttnn::Tensor& matmul_w0_w1_tensor,
        const ttnn::Tensor& matmul_w2_tensor,
        uint32_t layer_id,
        uint32_t output_height_shard_dim,
        uint32_t output_width_shard_dim,
        const std::vector<ttnn::CoreCoord>& output_shard_cores,
        const std::optional<uint32_t>& cluster_axis);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto moe_compute = ttnn::
    register_operation<"ttnn::experimental::moe_compute", ttnn::operations::experimental::ccl::ExecuteMoECompute>();

}  // namespace experimental
}  // namespace ttnn
