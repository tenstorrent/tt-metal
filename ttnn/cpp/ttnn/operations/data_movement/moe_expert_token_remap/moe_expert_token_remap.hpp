// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "device/moe_expert_token_remap_device_operation.hpp"

namespace ttnn {
namespace operations::data_movement {

struct ExecuteMoeExpertTokenRemap {
    static constexpr auto REDUCTION_SIZE = MoeExpertTokenRemapDeviceOperation::REDUCTION_SIZE;
    static std::vector<ttnn::Tensor> invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& expert_mapping_tensor,
        const ttnn::Tensor& expert_metadata_tensor,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<ttnn::Tensor>& optional_output_tensor = std::nullopt,
        const std::optional<ttnn::Tensor>& optional_reduced_tensor = std::nullopt,
        uint32_t reduction_size = REDUCTION_SIZE);
};

}  // namespace operations::data_movement

constexpr auto moe_expert_token_remap = ttnn::
    register_operation<"ttnn::moe_expert_token_remap", ttnn::operations::data_movement::ExecuteMoeExpertTokenRemap>();

}  // namespace ttnn
