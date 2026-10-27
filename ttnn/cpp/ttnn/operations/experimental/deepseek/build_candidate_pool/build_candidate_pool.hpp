// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::deepseek {

// Compacts the candidate blocks a ``fused_lightning_select_kv`` candidate source chose into a pool
// that later layers select from with the same op (identity page table, ``num_keys_tensor =
// pool_count``, ``row_table = pool_rows`` and the source's kv cache).
//
//     pool_keys[j] = key_cache at logical key candidate_blocks[j / cbs] * cbs + j % cbs
//     pool_rows[j] = kv cache row of the same logical key
//
// both through ``page_table_tensor``, which the kv cache shares. Only the first ``pool_count``
// entries (rounded up to whole candidate blocks) are written; the rest keep their previous contents.
//
// Args:
//   candidate_blocks:  ``[1, candidate_topk_blocks]`` INT32 kept blocks, ascending.
//   pool_count:        ``[1]`` INT32 number of valid keys in them.
//   key_cache:         bfloat16 TILE DRAM ``[num_blocks, 1, block_size, D]`` index key cache.
//   page_table_tensor: ``[1, max_blocks_per_user]`` INT32.
//   candidate_block_size: keys per candidate block, 2, 4, 8 or 16.
//   pool_keys_output, pool_rows_output: optional preallocated outputs.
//   page_table_row_tensor: optional ``[1]`` INT32 row of a ``[num_sessions, max_blocks_per_user]``
//                      page table to read, on device, so one traced call serves any session.
//
// Returns: ``[pool_keys, pool_rows]``: ``[candidate_topk_blocks * cbs / block_size, 1, block_size, D]``
// TILE with the key cache's dtype and ``[1, candidate_topk_blocks * cbs]`` INT32 row-major, both DRAM
// interleaved.
std::vector<Tensor> build_candidate_pool(
    const Tensor& candidate_blocks,
    const Tensor& pool_count,
    const Tensor& key_cache,
    const Tensor& page_table_tensor,
    uint32_t candidate_block_size,
    const std::optional<Tensor>& pool_keys_output = std::nullopt,
    const std::optional<Tensor>& pool_rows_output = std::nullopt,
    const std::optional<Tensor>& page_table_row_tensor = std::nullopt);

}  // namespace ttnn::experimental::deepseek
