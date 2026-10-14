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
//   output_tensor:   optional preallocated kv output ``[1, Hkv, rows, Dh]``, row-major
//                    interleaved, same dtype as ``kv_cache`` (e.g. a CSA layer's combined
//                    sliding-ring + compressed buffer). Rows outside the selected range are
//                    left untouched. ``memory_config`` is ignored when it is given.
//   output_row_offset: output row that selected row 0 is written to, e.g. ``sliding_window``
//                    to skip the ring. Needs ``output_tensor``; rows
//                    ``[output_row_offset, output_row_offset + k)`` must fit in it.
//   new_kv_row:      optional ``[1, 1, 1, Dh]`` row-major row, same dtype as ``kv_cache``,
//                    written into ``kv_cache`` at logical row ``new_kv_row_index`` (through
//                    ``page_table_tensor``) before any row is gathered, so a key closing at
//                    this step can be selected in the same call.
//   new_kv_row_index: ``[1]`` INT32 logical row of ``new_kv_row``. Given together with it.
//   num_keys_tensor: optional ``[1]`` INT32 number of selectable keys, replacing
//                    ``(cur_pos + 1) / 4``. A candidate-pool consumer passes the source's pool count.
//   candidate_topk_blocks, candidate_block_size: candidate source mode when both are > 0. Besides
//                    its own top-k, the op scores each block of ``candidate_block_size`` keys by its
//                    best key and keeps the ``candidate_topk_blocks`` best blocks, always including
//                    the block of the newest key.
//   candidate_blocks_output, pool_count_output: optional preallocated candidate source outputs.
//   num_parts:       key split across devices when > 1. The page blocks holding the selectable keys
//                    split into ``num_parts`` contiguous ranges; this device scores range
//                    ``part_index_tensor[0]`` and returns its local top-k as candidates instead of
//                    gathering rows. Gathering every device's candidates and passing them to
//                    ``merge_select_kv`` gives the rows the unsplit op would gather.
//   part_index_tensor: ``[1]`` INT32 part this device scores. Given exactly when ``num_parts`` > 1.
//   row_table:       optional ``[1, max_blocks_per_user * block_size]`` INT32 ``kv_cache`` row of each
//                    key, replacing the page-table mapping for the gather. A candidate-pool consumer
//                    passes the pool's rows so it gathers straight from the source's ``kv_cache``.
//                    ``new_kv_row_index`` is then a ``kv_cache`` row, since ``page_table_tensor``
//                    maps the pool.
//
// Returns: ``[kv_rows, scores]``, plus ``[candidate_blocks, pool_count]`` in candidate source mode.
//   kv_rows: selected KV rows, ``[1, Hkv, k, Dh]``, same dtype and layout as ``kv_cache``,
//            or ``output_tensor`` when given. With ``num_parts`` > 1, the candidates instead:
//            ``[1, 1, 1, 2k + 16]`` INT32, the selected keys' ``kv_cache`` rows in key order, then
//            their order keys (fp32 score bits mapped so unsigned order is score order), then the
//            number selected. Unused entries are zero.
//   scores:  index scores, ``[1, 1, 1, max_blocks_per_user * block_size]`` fp32 ROW_MAJOR. Only the
//            first ``div_up((cur_pos + 1) / 4, block_size) * block_size`` entries are written.
//   candidate_blocks: ``[1, candidate_topk_blocks]`` INT32, the kept blocks in ascending order in
//            the first ``min(candidate_topk_blocks, number of blocks)`` entries, zero after them.
//   pool_count: ``[1]`` INT32, the keys the kept blocks hold (the newest block may be partial).
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
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    uint32_t output_row_offset = 0,
    const std::optional<Tensor>& new_kv_row = std::nullopt,
    const std::optional<Tensor>& new_kv_row_index = std::nullopt,
    const std::optional<Tensor>& num_keys_tensor = std::nullopt,
    uint32_t candidate_topk_blocks = 0,
    uint32_t candidate_block_size = 0,
    const std::optional<Tensor>& candidate_blocks_output = std::nullopt,
    const std::optional<Tensor>& pool_count_output = std::nullopt,
    uint32_t num_parts = 1,
    const std::optional<Tensor>& part_index_tensor = std::nullopt,
    const std::optional<Tensor>& row_table = std::nullopt);

}  // namespace ttnn::experimental::deepseek
