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
 * - Skips non-local experts using dispatch table + indices (saves ~75% compute on TP4)
 * - Outputs result ready for reduce_scatter
 *
 * Input shapes:
 * - combine_output: [1, dispatch_group_size, seq_len, num_experts_per_tok, emb_dim] (ROW_MAJOR)
 * - weights: [dispatch_group_size, seq_len, num_experts_per_tok] (ROW_MAJOR or TILE_LAYOUT)
 * - indices: [dispatch_group_size, seq_len, num_experts_per_tok] (ROW_MAJOR, INT32)
 * - expert_dispatch_table: [num_routed_experts] (ROW_MAJOR, INT32) — sharded per dispatch group
 *
 * Output shape:
 * - [dispatch_group_size, seq_len, emb_dim] (TILE_LAYOUT, ready for reduce_scatter)
 */
ttnn::Tensor post_combine_reduce(
    const ttnn::Tensor& combine_output,
    const ttnn::Tensor& weights,
    const ttnn::Tensor& indices,
    const ttnn::Tensor& expert_dispatch_table,
    uint32_t expert_dim = 3,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::post_combine_reduce
