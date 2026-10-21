// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <cstdint>
#include <optional>

namespace ttnn::transformer {

// DeepSeek V4 lightning-indexer top-k for a prefill chunk, Blackhole single-chip. Fuses the indexer score and the
// top-k so neither the [T, E] score matrix nor a capacity-sized buffer is materialized:
//   score[s, t] = sum_h relu(q[h, s] . idx_keys[t]) * weights[s, h]
// over the entries t < (start + s + 1) / compress_ratio query s can see, of which the top 512 are returned.
//   q         [1, H, T, D] bf16 TILE, H and T multiples of 32 (post-RoPE indexer queries)
//   weights   [1, 1, T, H] bf16 TILE, the per-head gates with every scale folded in
//   idx_keys  [1, 1, E, D] bf16 ROW_MAJOR indexer keys; only the visible prefix is read, and a longer E reuses
//             the program
//   start     1-element uint32/int32 ROW_MAJOR device tensor: the absolute position of query row 0. Read on
//             device, so a captured trace replays at a new position.
// Returns [1, 1, T, 512] uint32 ROW_MAJOR entry indices in no particular order; a query that sees fewer than 512
// entries fills the remaining slots with 0xFFFFFFFF. Visible entries must lie within E.
ttnn::Tensor deepseek_v4_prefill_indexer_select(
    const ttnn::Tensor& q,
    const ttnn::Tensor& weights,
    const ttnn::Tensor& idx_keys,
    const ttnn::Tensor& start,
    uint32_t compress_ratio,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn::transformer
