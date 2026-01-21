// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operation.hpp"

#include "moe_expert_token_remap.hpp"

namespace ttnn::operations::data_movement {

std::vector<ttnn::Tensor> ExecuteMoeExpertTokenRemap::invoke(
    const ttnn::Tensor& topk_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    const ttnn::Tensor& expert_metadata_tensor,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    const std::optional<ttnn::Tensor>& optional_reduced_tensor,
    const uint32_t reduction_size) {
    return ttnn::prim::moe_expert_token_remap(
        topk_tensor,
        expert_mapping_tensor,
        expert_metadata_tensor,
        memory_config,
        optional_output_tensor,
        optional_reduced_tensor,
        reduction_size);
}

}  // namespace ttnn::operations::data_movement
