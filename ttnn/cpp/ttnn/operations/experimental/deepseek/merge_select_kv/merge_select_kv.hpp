// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::deepseek {

// Second half of a key-split ``fused_lightning_select_kv``: merges the per-device candidates into
// the global top-k and gathers those rows from this device's ``kv_cache``.
//
// ``candidates`` is ``[1, 1, num_parts, 2k + 16 + 2 * candidate_topk_blocks]`` INT32, the
// ``num_parts > 1`` outputs of ``fused_lightning_select_kv`` concatenated along dim 2 in part order
// (``ttnn.all_gather``). The merge keeps the k largest order keys over every part, ties to the
// earlier part and then the earlier entry, i.e. to the lower key, which is the unsplit op's
// selection. Rows are written in part order and then entry order, i.e. ascending key order like the
// unsplit op.
//
// Args:
//   candidates:   ``[1, 1, num_parts, 2k + 16 + 2 * candidate_topk_blocks]`` INT32 row-major interleaved.
//   kv_cache:     row-major interleaved ``[num_blocks, 1, block_size, Dh]``; the candidates' rows index it.
//   k:            number of selected rows.
//
// Keyword Args:
//   memory_config:     output memory config, interleaved. Defaults to interleaved L1.
//   output_tensor:     optional preallocated ``[1, 1, rows, Dh]`` row-major interleaved output with
//                      ``kv_cache``'s dtype; only rows ``[output_row_offset, output_row_offset + k)``
//                      are written.
//   output_row_offset: output row of selected row 0. Needs ``output_tensor``.
//   candidate_topk_blocks, candidate_block_size: candidate source mode when both are > 0, for the
//                      candidates of a candidate source split. The ``candidate_topk_blocks`` best
//                      blocks over every part, ties to the lower block, go to
//                      ``candidate_blocks_output`` in ascending order (zero past the last), and the
//                      keys they hold to ``pool_count_output``: the unsplit op's outputs.
//   candidate_blocks_output: preallocated ``[1, candidate_topk_blocks]`` INT32 row-major interleaved.
//                      Given exactly in candidate source mode.
//   pool_count_output: preallocated ``[1]`` INT32 row-major interleaved. Given exactly in candidate
//                      source mode.
//
// Returns: the selected rows, ``[1, 1, k, Dh]``, or ``output_tensor``. With fewer than k candidates
// in total, only that many rows are written.
Tensor merge_select_kv(
    const Tensor& candidates,
    const Tensor& kv_cache,
    uint32_t k,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    uint32_t output_row_offset = 0,
    uint32_t candidate_topk_blocks = 0,
    uint32_t candidate_block_size = 0,
    const std::optional<Tensor>& candidate_blocks_output = std::nullopt,
    const std::optional<Tensor>& pool_count_output = std::nullopt);

}  // namespace ttnn::experimental::deepseek
