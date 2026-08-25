// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::transformer {

/**
 * Fused T=1 (decode step) Gated Delta Rule forward.
 *
 * Implements the recurrent decode graph of
 * `recurrent_gated_delta_rule_decode_ttnn` in ONE reader/compute/writer
 * program, one Tensix core per (B*H) head, fp32 accumulation:
 *
 *   qn = l2norm(q) * scale ; kn = l2norm(k)
 *   h  = initial_state * exp(g)
 *   v_read = kn @ h ; delta = v - v_read
 *   h += (kn)^T @ (beta * delta)          (rank-1 write)
 *   o  = qn @ h
 *
 *   q, k          [B, 1, H, K]   TILE, bf16 or fp32 (all one dtype)
 *   v             [B, 1, H, V]   (H heads only; no GQA)
 *   beta, g       [B, 1, H]      g is log-space decay
 *   initial_state [B, H, K, V]   TILE, same dtype (absent => zeros)
 *
 * Returns:
 *   o          [B, 1, H, V] ROW_MAJOR — each head owns its [V] stick as a
 *               whole DRAM page, so the writer only issues full-page writes
 *               (sub-page writes do not land on this stack). Feed through
 *               ttnn.to_layout(o, TILE_LAYOUT) when the next op needs TILE.
 *   new_state  [B, H, K, V] TILE
 *
 * inplace_state=True (requires initial_state): the writer stores new_state
 * into initial_state's buffer and initial_state is returned unchanged as
 * new_state — the fused equivalent of the python `_inplace` variant's
 * copy-back into a preallocated state buffer (trace-safe addresses, no
 * allocation).
 */
std::tuple<ttnn::Tensor, ttnn::Tensor> decode_gated_delta_rule(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& beta,
    const ttnn::Tensor& g,
    std::optional<float> scale = std::nullopt,
    const std::optional<ttnn::Tensor>& initial_state = std::nullopt,
    bool inplace_state = false,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::transformer
