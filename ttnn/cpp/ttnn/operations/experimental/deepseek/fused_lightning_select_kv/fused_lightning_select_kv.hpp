// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::deepseek {

// Fused lightning indexer select for DeepSeek V4-Flash CSA decode.
//
// Replaces ``indexer_score_dsa`` + ``topk_large_indices`` plus the gather that
// ``sparse_sdpa`` does from the index list, and returns the selected attention
// KV rows so a dense SDPA can consume them directly (K == V).
//
//     scores  = indexer_score_dsa(query, key_cache, head_weights, chunk_start_idx = T - Sq)
//     indices = topk_large_indices(scores, k, valid_length_tensor)
//     out     = kv_cache gathered at indices through page_table          [B, Hkv, k, Dh]
//
// ``query`` is TILE ``[B, Hi, Sq, D]``, ``key_cache`` is the indexer key cache
// TILE ``[B, 1, T, D]``, ``head_weights`` is TILE ``[B, 1, Sq, Hi]``.
//
// ``kv_cache`` is a paged block pool ``[num_blocks, Hkv, block_size, Dh]``, read
// through ``page_table_tensor`` ``[B, max_blocks_per_user]`` INT32, the same
// layout ``paged_scaled_dot_product_attention_decode`` takes. ``cur_pos_tensor``
// ``[B]`` INT32 is each user's current (inclusive) position on that axis; rows
// past it are never read.
//
// The output keeps ``kv_cache``'s dtype and layout; its shape is ``[B, Hkv, k, Dh]``.
//
// The device kernel is not implemented. This entry point only fixes the
// host-side contract (arguments, validation, output spec).
//
// Args:
//   query:             indexer query, ``[B, Hi, Sq, D]``.
//   key_cache:         indexer key cache, ``[B, 1, T, D]``.
//   head_weights:      folded head scales, ``[B, 1, Sq, Hi]``.
//   kv_cache:          paged attention KV pool, ``[num_blocks, Hkv, block_size, Dh]``.
//   page_table_tensor: per-user block ids, ``[B, max_blocks_per_user]`` INT32.
//   cur_pos_tensor:    per-user current position, ``[B]`` INT32.
//   k:                 number of selected rows.
//
// Keyword Args:
//   valid_length_tensor: optional 1-element uint32 tensor. Score columns at or
//                    past this length are not selectable, matching ``topk_large_indices``.
//   memory_config: output memory config. Defaults to interleaved DRAM.
//   compute_kernel_config: compute settings for the score. Defaults to HiFi4
//                    with fp32 destination accumulation.
//
// Returns: selected KV rows, ``[B, Hkv, k, Dh]``, same dtype and layout as ``kv_cache``.
Tensor fused_lightning_select_kv(
    const Tensor& query,
    const Tensor& key_cache,
    const Tensor& head_weights,
    const Tensor& kv_cache,
    const Tensor& page_table_tensor,
    const Tensor& cur_pos_tensor,
    uint32_t k,
    const std::optional<Tensor>& valid_length_tensor = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn::experimental::deepseek
