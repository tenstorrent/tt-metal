// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::deepseek {

// Single-token MQA decode attention for DeepSeek V4.1 (one shared K == V head, per-head sinks).
// Attends the valid rows of two KV segments: the sliding-window ring ``kv`` and the CSA top-k
// ``selected_kv``, so the compressed layers need no concatenation of the two. Each chip runs its own
// heads (TP), split into groups of 8 per core; the chunks of both segments are spread over
// ``num_s_blocks`` cores per head group and merged with a tree reduction (flash decoding).
//
// Args:
//   q:       ``[1, 1, heads, head_dim]`` bfloat16 row-major; heads a multiple of 8.
//   kv:      ``[1, 1, rows, head_dim]`` TILE bfloat16 / bfloat8_b; rows a multiple of k_chunk_size.
//   lengths: row-major int32 whose first row starts with ``[ring_valid, selected_valid]`` and, optionally,
//            ``ring_first_row`` and then ``selected_first_row``: the attended rows are
//            ``kv[ring_first_row:ring_first_row + ring_valid]`` and
//            ``selected_kv[offset + selected_first_row:offset + selected_first_row + selected_valid]``.
//            ring_valid must be at least 1, ring_first_row and selected_first_row multiples of 32. Read on
//            device, so it can change under a trace.
//
// Keyword Args:
//   selected_kv:         ``[1, 1, rows, head_dim]`` TILE, kv's dtype.
//   selected_row_offset: row of selected_kv holding selected row 0, a multiple of 32.
//   ring_rows:           most ring rows attended (one session's ring in a shared ``kv``), a multiple of
//                        k_chunk_size; 0 for all of kv's rows.
//   selected_rows:       most selected rows attended (one session's rows in a shared ``selected_kv``), a
//                        multiple of k_chunk_size; 0 for all of selected_kv's rows past selected_row_offset.
//   attention_sink:      TILE bfloat16 ``[heads, 32]`` with ``sink / scale`` in column 0.
//   scale:               softmax scale, default ``head_dim ** -0.5``.
//   k_chunk_size:        rows per chunk, a multiple of 64.
//   num_s_blocks:        cores per head group.
//   memory_config:       output memory config, default q's.
//   cos:                 optional TILE bfloat16 ``[1, 1, >=1, rope_dim]`` RoPE cosines; when set with
//                        ``sin``, applies interleaved RoPE in place to the trailing ``rope_dim``
//                        columns (row 0 of ``cos``/``sin``) before writing the output.
//   sin:                 optional TILE bfloat16 ``[1, 1, >=1, rope_dim]`` RoPE sines (pass ``-sin``
//                        for inverse RoPE).
//
//   new_kv, num_rows, dense_selected, row_groups: a block of ``num_rows`` (2, 4 or 8) query rows, see below.
//
// Returns: ``[1, 1, heads, head_dim]`` bfloat16 row-major.
//
// Block mode (``num_rows`` R > 1, a DSpark verify block at positions p .. p + R - 1, causal inside the block):
//   q:       ``[1, 1, heads * R, head_dim]`` bfloat16 row-major, row ``h * R + j`` (head-major).
//   kv:      the ring, holding positions <= p - 1 at row ``ring_first_row + position % ring_capacity``.
//   new_kv:  TILE ``[1, 1, 32, head_dim]`` with kv's dtype: rows 0 .. R - 1 are the block's new kv rows (the
//            other rows must be finite; they are masked).
//   lengths: row-major int32 ``[1, 1, 1, 4 + R]``: ``[p, ring_first_row, selected_first_row, 0, n_0 .. n_{R-1}]``.
//            Row j attends the ring slots of positions ``max(0, p + j - (ring_capacity - 1)) .. p - 1``, new kv rows
//            ``0 .. j``, and ``n_j`` selected rows: ``selected_kv`` rows from
//            ``selected_row_offset + selected_first_row + j * selected_capacity`` (``dense_selected`` false) or from
//            ``selected_row_offset + selected_first_row`` (``dense_selected`` true). ``selected_capacity`` is
//            ``selected_rows`` (or, when 0, the rows past the offset / R per-row, all of them dense).
//   cos/sin: TILE ``[1, 1, R, rope_dim]`` (tile ``(R, 32)`` or 32x32), row j for block row j. Required.
//   row_groups: the rows run on row_groups groups of num_s_blocks x head-group cores (group rg attends rows rg,
//            rg + row_groups, ...; a chunk shared by rows is read once per group); 0 picks the most that divides R
//            and fits the grid's height.
//   Returns TILE ``(R, 32)`` ``[1, 1, R, heads * head_dim]`` bfloat16; default memory: one L1 shard on q's first
//   core when q is sharded, else q's memory config.
Tensor flash_mqa_decode(
    const Tensor& q,
    const Tensor& kv,
    const Tensor& lengths,
    const std::optional<Tensor>& selected_kv = std::nullopt,
    uint32_t selected_row_offset = 0,
    uint32_t ring_rows = 0,
    uint32_t selected_rows = 0,
    const std::optional<Tensor>& attention_sink = std::nullopt,
    std::optional<float> scale = std::nullopt,
    uint32_t k_chunk_size = 64,
    uint32_t num_s_blocks = 8,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    const std::optional<Tensor>& cos = std::nullopt,
    const std::optional<Tensor>& sin = std::nullopt,
    const std::optional<Tensor>& new_kv = std::nullopt,
    uint32_t num_rows = 1,
    bool dense_selected = false,
    uint32_t row_groups = 0);

}  // namespace ttnn::experimental::deepseek
