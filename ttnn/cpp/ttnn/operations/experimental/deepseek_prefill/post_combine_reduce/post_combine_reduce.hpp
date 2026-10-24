// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::post_combine_reduce {

/**
 * Fused post-combine reduce operation for DeepSeek / GPT-OSS MoE.
 *
 * Replaces the inefficient sequence of:
 * 1. ttnn.to_layout() - ROW_MAJOR -> TILE_LAYOUT with fillpad (8->32 experts)
 * 2. ttnn.mul() - broadcast weights across embedding dimension
 * 3. ttnn.sum() - reduce over expert dimension
 *
 * With a single fused kernel that:
 * - Reads ROW_MAJOR combine output directly (no padding overhead)
 * - Reads and multiplies only the active (token, slot) rows (saves ~75% of reads on TP4)
 * - Performs broadcast multiply + reduce in one pass
 * - Outputs result ready for reduce_scatter
 *
 * Two expert-skip strategies are supported, selected by whether the optional
 * indices + expert_dispatch_table tensors are supplied:
 *
 *   DeepSeek path — BOTH indices and expert_dispatch_table provided:
 *     A slot is active when its expert id maps to a chip (not -1, non-local to
 *     this dispatch group) in the dispatch table. Inactive slots are never read,
 *     so the upstream combine op does not need to zero them.
 *
 *   GPT-OSS path — NEITHER indices nor expert_dispatch_table provided:
 *     A slot is active when its routing weight is non-zero. This requires the
 *     upstream router to have zeroed routing weights for non-local experts
 *     (which is cheaper than materialising a dispatch table when weights are
 *     already per-token).
 *
 *   A token with no active slot produces exact zeros.
 *   Supplying exactly one of the two raises a TT_FATAL.
 *
 * Input shapes:
 * - combine_output: [1, dispatch_group_size, seq_len, num_experts_per_tok, emb_dim] (ROW_MAJOR)
 * - weights: [dispatch_group_size, seq_len, num_experts_per_tok] (ROW_MAJOR or TILE_LAYOUT)
 * - indices (optional): [dispatch_group_size, seq_len, num_experts_per_tok] (ROW_MAJOR, INT32)
 * - expert_dispatch_table (optional): [num_routed_experts] (ROW_MAJOR, INT32) — sharded per dispatch group
 *
 * Output shape:
 * - [dispatch_group_size, seq_len, emb_dim] (TILE_LAYOUT, ready for reduce_scatter)
 */
ttnn::Tensor post_combine_reduce(
    const ttnn::Tensor& combine_output,
    const ttnn::Tensor& weights,
    const std::optional<ttnn::Tensor>& indices = std::nullopt,
    const std::optional<ttnn::Tensor>& expert_dispatch_table = std::nullopt,
    uint32_t expert_dim = 3,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::post_combine_reduce
