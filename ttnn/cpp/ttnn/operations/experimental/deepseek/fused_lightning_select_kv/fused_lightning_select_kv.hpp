// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

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
//     scores[t] = sum_h ReLU(q_h . key_cache[t]) * w_h
//     indices   = topk_large_indices(scores, k, valid_length_tensor)
//     out       = kv_cache gathered at indices through page_table        [1, Hkv, k, Dh]
//
// Decode only: one user (B == 1) and one query token (Sq == 1).
//
// ``query`` ``[num_cores, Hi, 1, D]`` and ``head_weights`` ``[num_cores, 1, 1, Hi]``
// are ROW_MAJOR HEIGHT_SHARDED in L1 with one full replica per core (shards
// ``[Hi, D]`` and ``[1, Hi]``), the ``matmul_decode`` rm_hs layout. The op runs on
// their shared shard grid. ``key_cache`` is the indexer key cache ``[1, 1, T, D]``.
//
// ``kv_cache`` is a paged block pool ``[num_blocks, Hkv, block_size, Dh]``, read
// through ``page_table_tensor`` ``[1, max_blocks_per_user]`` INT32, the same
// layout ``paged_scaled_dot_product_attention_decode`` takes. ``cur_pos_tensor``
// ``[1]`` INT32 is the user's current (inclusive) position on that axis; rows
// past it are never read.
//
// The output keeps ``kv_cache``'s dtype and layout; its shape is ``[1, Hkv, k, Dh]``.
//
// The device kernel is not implemented. This entry point only fixes the
// host-side contract (arguments, validation, output spec).
//
// Args:
//   query:             indexer query, ``[num_cores, Hi, 1, D]`` replicated rm_hs.
//   key_cache:         indexer key cache, ``[1, 1, T, D]``.
//   head_weights:      folded head scales, ``[num_cores, 1, 1, Hi]`` replicated rm_hs.
//   kv_cache:          paged attention KV pool, ``[num_blocks, Hkv, block_size, Dh]``.
//   page_table_tensor: block ids, ``[1, max_blocks_per_user]`` INT32.
//   cur_pos_tensor:    current position, ``[1]`` INT32.
//   k:                 number of selected rows.
//
// Keyword Args:
//   valid_length_tensor: optional 1-element uint32 tensor. Score columns at or
//                    past this length are not selectable, matching ``topk_large_indices``.
//   memory_config: output memory config. Defaults to interleaved DRAM.
//   compute_kernel_config: compute settings for the score. Defaults to HiFi4
//                    with fp32 destination accumulation.
//
// Returns: ``[kv_rows, scores]``.
//   kv_rows: selected KV rows, ``[1, Hkv, k, Dh]``, same dtype and layout as ``kv_cache``.
//   scores:  index scores, ``[1, 1, 1, max_blocks_per_user * block_size]`` fp32 ROW_MAJOR. Only the
//            first ``div_up((cur_pos + 1) / 4, block_size) * block_size`` entries are written.
std::vector<Tensor> fused_lightning_select_kv(
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
