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
// ``candidates`` is ``[1, 1, num_parts, 2k + 16]`` INT32, the ``num_parts > 1`` outputs of
// ``fused_lightning_select_kv`` concatenated along dim 2 in part order (``ttnn.all_gather``). The
// merge keeps the k largest order keys over every part, ties to the earlier part and then the
// earlier entry, i.e. to the lower key, which is the unsplit op's selection. Rows are written in
// part order and then entry order, i.e. ascending key order like the unsplit op.
//
// Args:
//   candidates:   ``[1, 1, num_parts, 2k + 16]`` INT32 row-major interleaved.
//   kv_cache:     row-major interleaved ``[num_blocks, 1, block_size, Dh]``; the candidates' rows index it.
//   k:            number of selected rows.
//
// Keyword Args:
//   memory_config:     output memory config, interleaved. Defaults to interleaved L1.
//   output_tensor:     optional preallocated ``[1, 1, rows, Dh]`` row-major interleaved output with
//                      ``kv_cache``'s dtype; only rows ``[output_row_offset, output_row_offset + k)``
//                      are written.
//   output_row_offset: output row of selected row 0. Needs ``output_tensor``.
//
// Returns: the selected rows, ``[1, 1, k, Dh]``, or ``output_tensor``. With fewer than k candidates
// in total, only that many rows are written.
Tensor merge_select_kv(
    const Tensor& candidates,
    const Tensor& kv_cache,
    uint32_t k,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    uint32_t output_row_offset = 0);

}  // namespace ttnn::experimental::deepseek
