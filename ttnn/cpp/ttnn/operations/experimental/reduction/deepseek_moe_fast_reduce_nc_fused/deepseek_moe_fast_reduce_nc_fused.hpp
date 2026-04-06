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
// input_tensor  : [experts_k, 1, tokens, hidden_size]  TILE layout, L1
// scores_tensor : optional [tokens, 1, seq, experts_k] ROW_MAJOR layout, DRAM.
//                 If std::nullopt, delegates to deepseek_moe_fast_reduce_nc (no fused score multiply).
//
// Returns split_size-wide output slices [1, 1, tokens, split_size] × (hidden_size/split_size)
std::vector<ttnn::Tensor> deepseek_moe_fast_reduce_nc_fused(
    const ttnn::Tensor& input_tensor,
    int32_t reduce_dim,
    uint64_t split_size,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    const std::optional<ttnn::Tensor>& scores_tensor,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::experimental::reduction
