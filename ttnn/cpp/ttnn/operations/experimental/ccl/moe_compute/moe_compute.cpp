// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>

#include "moe_compute.hpp"
#include "device/moe_compute_device_operation.hpp"

namespace ttnn {
namespace experimental {

std::vector<ttnn::Tensor> moe_compute(
    const ttnn::Tensor& tilize_input_tensor,
    const ttnn::Tensor& tilize_expert_indices_tensor,
    const ttnn::Tensor& tilize_expert_scores_tensor,
    const ttnn::Tensor& tilize_expert_mapping_tensor,
    const ttnn::Tensor& matmul_w0_w1_tensor,
    const ttnn::Tensor& matmul_w2_tensor,
    const uint32_t layer_id,
    const uint32_t output_height_shard_dim,
    const uint32_t output_width_shard_dim,
    const std::optional<uint32_t>& cluster_axis) {
    return ttnn::prim::moe_compute(
        tilize_input_tensor,
        tilize_expert_indices_tensor,
        tilize_expert_scores_tensor,
        tilize_expert_mapping_tensor,
        matmul_w0_w1_tensor,
        matmul_w2_tensor,
        layer_id,
        output_height_shard_dim,
        output_width_shard_dim,
        cluster_axis);
}

std::vector<ttnn::CoreCoord> get_moe_combine_cores(ttnn::MeshDevice* mesh_device) {
    return ttnn::prim::get_moe_combine_cores(mesh_device);
};
}  // namespace experimental
}  // namespace ttnn
