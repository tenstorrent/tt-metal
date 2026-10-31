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
//            ``ring_first_row``: the attended rows are ``kv[ring_first_row:ring_first_row + ring_valid]``
//            and ``selected_kv[offset:offset + selected_valid]``. ring_valid must be at least 1 and
//            ring_first_row a multiple of 32. Read on device, so it can change under a trace.
//
// Keyword Args:
//   selected_kv:         ``[1, 1, rows, head_dim]`` TILE, kv's dtype.
//   selected_row_offset: row of selected_kv holding selected row 0, a multiple of 32.
//   ring_rows:           most ring rows attended (one session's ring in a shared ``kv``), a multiple of
//                        k_chunk_size; 0 for all of kv's rows.
//   attention_sink:      TILE bfloat16 ``[heads, 32]`` with ``sink / scale`` in column 0.
//   scale:               softmax scale, default ``head_dim ** -0.5``.
//   k_chunk_size:        rows per chunk, a multiple of 64.
//   num_s_blocks:        cores per head group.
//   memory_config:       output memory config, default q's.
//
// Returns: ``[1, 1, heads, head_dim]`` bfloat16 row-major.
Tensor flash_mqa_decode(
    const Tensor& q,
    const Tensor& kv,
    const Tensor& lengths,
    const std::optional<Tensor>& selected_kv = std::nullopt,
    uint32_t selected_row_offset = 0,
    uint32_t ring_rows = 0,
    const std::optional<Tensor>& attention_sink = std::nullopt,
    std::optional<float> scale = std::nullopt,
    uint32_t k_chunk_size = 64,
    uint32_t num_s_blocks = 8,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::experimental::deepseek
