// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::post_combine_reduce {

/**
 * Fused post-combine reduce operation for DeepSeek MoE.
 *
 * Replaces the inefficient sequence of:
 * 1. ttnn.to_layout() - ROW_MAJOR -> TILE_LAYOUT with fillpad (8->32 experts)
 * 2. ttnn.mul() - broadcast weights across embedding dimension
 * 3. ttnn.sum() - reduce over expert dimension
 *
 * With a single fused kernel that:
 * - Reads ROW_MAJOR combine output directly (no padding overhead)
 * - Performs broadcast multiply + reduce in one pass
 * - Outputs result ready for reduce_scatter
 *
 * Input shapes:
 * - combine_output: [1, dispatch_group_size, seq_len, num_experts_per_tok, emb_dim] (ROW_MAJOR)
 * - weights: [dispatch_group_size, seq_len, num_experts_per_tok] (ROW_MAJOR or TILE_LAYOUT)
 *
 * Output shape:
 * - [dispatch_group_size, seq_len, emb_dim] (TILE_LAYOUT, ready for reduce_scatter)
 */
ttnn::Tensor post_combine_reduce(
    const ttnn::Tensor& combine_output,  // MoE combine output (ROW_MAJOR)
    const ttnn::Tensor& weights,         // Gate weights for broadcast multiply
    uint32_t expert_dim = 3,             // Dimension to reduce over (default: 3)
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::post_combine_reduce
