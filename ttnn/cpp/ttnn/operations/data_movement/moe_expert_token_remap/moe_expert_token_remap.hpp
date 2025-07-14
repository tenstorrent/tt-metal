// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::data_movement {

struct ExecuteMoeExpertTokenRemap {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& expert_mapping_tensor,
        const ttnn::Tensor& expert_metadata_tensor,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<ttnn::Tensor>& optional_output_tensor = std::nullopt);
};

}  // namespace operations::data_movement

constexpr auto moe_token_expert_remap = ttnn::
    register_operation<"ttnn::moe_token_expert_remap", ttnn::operations::data_movement::ExecuteMoeExpertTokenRemap>();

}  // namespace ttnn
