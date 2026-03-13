// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/types.hpp"

namespace ttnn {
namespace experimental {

std::vector<ttnn::Tensor> moe_compute(
    const ttnn::Tensor& tilize_input_tensor,
    const ttnn::Tensor& tilize_expert_indices_tensor,
    const ttnn::Tensor& tilize_expert_scores_tensor,
    const ttnn::Tensor& tilize_expert_mapping_tensor,
    const ttnn::Tensor& matmul_w0_w1_tensor,
    const ttnn::Tensor& matmul_w2_tensor,
    uint32_t layer_id,
    uint32_t output_height_shard_dim,
    uint32_t output_width_shard_dim,
    const std::optional<uint32_t>& cluster_axis);

std::vector<ttnn::CoreCoord> get_moe_combine_cores(ttnn::MeshDevice* mesh_device);

}  // namespace experimental
}  // namespace ttnn
