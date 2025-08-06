// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>

#include "ttnn/common/queue_id.hpp"
#include "ttnn/run_operation.hpp"
#include "device/moe_expert_token_remap_device_operation.hpp"

#include "moe_expert_token_remap.hpp"

namespace ttnn::operations::data_movement {

ttnn::Tensor ExecuteMoeExpertTokenRemap::invoke(
    QueueId queue_id,
    const ttnn::Tensor& topk_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    const ttnn::Tensor& expert_metadata_tensor,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor) {
    return ttnn::prim::moe_expert_token_remap(
        topk_tensor, expert_mapping_tensor, expert_metadata_tensor, memory_config, optional_output_tensor);
}

}  // namespace ttnn::operations::data_movement
