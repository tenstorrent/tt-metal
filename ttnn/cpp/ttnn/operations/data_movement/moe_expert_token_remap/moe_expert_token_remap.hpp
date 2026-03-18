// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "device/moe_expert_token_remap_device_operation.hpp"

namespace ttnn {

std::vector<ttnn::Tensor> moe_expert_token_remap(
    const ttnn::Tensor& topk_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    const ttnn::Tensor& expert_metadata_tensor,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<ttnn::Tensor>& optional_reduced_tensor = std::nullopt,
    uint32_t reduction_size = ttnn::operations::data_movement::MoeExpertTokenRemapDeviceOperation::REDUCTION_SIZE);

}  // namespace ttnn
