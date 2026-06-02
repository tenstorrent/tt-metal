// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::reduction {

// Fused operation: permute(scores) + tilize + mul(activation, scores) + reduce_nc
// all in a single kernel launch, eliminating the large intermediate activation tensor.
//
// input_tensor          : [experts_k, 1, tokens, hidden_size]  TILE layout, L1
// expert_indices_tensor : per-token expert indices (matches all_to_all_dispatch convention)
// expert_mapping_tensor : expert-to-device mapping (matches all_to_all_dispatch convention)
// scores_tensor         : optional [tokens, 1, seq, experts_k] ROW_MAJOR layout, DRAM.
//                         If std::nullopt, delegates to deepseek_moe_fast_reduce_nc
//                         (no fused score multiply). In that fallback the expert_indices /
//                         expert_mapping / cluster_axis arguments are ignored.
// cluster_axis          : mesh axis (0 or 1) along which the expert mapping is laid out.
//
// Returns split_size-wide output slices [1, 1, tokens, split_size] × (hidden_size/split_size)
std::vector<ttnn::Tensor> deepseek_moe_fast_reduce_nc_fused(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_indices_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    int32_t reduce_dim,
    uint64_t split_size,
    uint32_t cluster_axis,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    const std::optional<ttnn::Tensor>& scores_tensor,
    uint32_t num_shared_experts = 0,
    float shared_expert_scale = 1.0f,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::experimental::reduction
