// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>

#include "moe.hpp"
#include "device/moe_device_operation.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteMoE::invoke(
    const ttnn::Tensor& tilize_input_tensor,
    const ttnn::Tensor& tilize_expert_indices_tensor,
    const ttnn::Tensor& tilize_expert_scores_tensor,
    const ttnn::Tensor& tilize_expert_mapping_tensor,
    const ttnn::Tensor& matmul_w0_w1_tensor,
    const ttnn::Tensor& matmul_w2_tensor,
    const uint32_t layer_id,
    const std::optional<uint32_t> cluster_axis) {
    // TODO: (GR) which tensor
    return ttnn::prim::moe(
               tilize_input_tensor,
               tilize_expert_indices_tensor,
               tilize_expert_scores_tensor,
               tilize_expert_mapping_tensor,
               matmul_w0_w1_tensor,
               matmul_w2_tensor,
               layer_id,
               cluster_axis)
        .at(0);
}

}  // namespace ttnn::operations::experimental::ccl
