// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::fused_experts_prefill {

// Routed-expert FFN for DeepSeek-V4-Flash prefill that consumes the DECODE weight layout in place and
// takes the same routing arguments as the decode `fused_experts`.
//
//   act_e = clamped_swiglu(x_t @ gate_up_e)   (silu(min(gate, limit)) * clamp(up, -limit, limit))
//   out_t = sum over the token's top_k experts e of  w[t, e] * (act_e @ down_e)
//
// with w the selected UNBIASED scores renormalised to sum to 1 (+ eps) and scaled by
// routed_scaling_factor. The routing (ids, weights, per-expert token lists) is computed on device: every
// one of the 120 worker cores runs the full routing over all T tokens and keeps the token lists of the
// experts it owns (expert e -> group e % 15 of 8 cores). Token rows are gathered from the row-major
// x_tok in DRAM, tilized on device and pushed through the FFN; the 8 cores of a group split the I dim
// (gate/up + SwiGLU) and the H dim (down) and exchange the SwiGLU activation with unicast writes.
//
// Every (token, expert) result is written, already weighted, to its own row of an intermediate
// [1, top_k, T, H] tensor (slot j of token t at row j * T + t) and the op returns the sum over slots.
//
// Args:
//   x_tok:           [1, 1, T, H] ROW_MAJOR BFLOAT16, DRAM interleaved; T a multiple of 32, at most 512.
//   routing_scores:  [1, 1, T, E] TILE BFLOAT16, the unbiased scores.
//   gate_up_weights: E tensors [H, 2I], decode layout (DRAM ND-sharded [H, 64]).
//   down_weights:    E tensors [I, H], decode layout (DRAM ND-sharded [I, 64]).
//   intermediate_size, swiglu_limit, top_k, routed_scaling_factor, routing_eps: as decode.
//   routing_indices: [1, 1, T, top_k] TILE UINT16 / BFLOAT16 selected expert ids (valid and unique per
//                    token, as a top-k produces them; an invalid / repeated id contributes nothing but
//                    leaves its partial row uninitialised).   -- or --
//   ranking_scores:  [1, 1, T, E] TILE BFLOAT16 ranked on device (top-k, lower id wins ties).
//
// Returns [1, 1, T, H] TILE BFLOAT16.
ttnn::Tensor fused_experts_prefill(
    const ttnn::Tensor& x_tok,
    const ttnn::Tensor& routing_scores,
    const std::vector<ttnn::Tensor>& gate_up_weights,
    const std::vector<ttnn::Tensor>& down_weights,
    uint32_t intermediate_size,
    float swiglu_limit,
    uint32_t top_k,
    float routed_scaling_factor,
    float routing_eps,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::Tensor>& routing_indices = std::nullopt,
    const std::optional<ttnn::Tensor>& ranking_scores = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::fused_experts_prefill

namespace ttnn {
using operations::experimental::deepseek_prefill::fused_experts_prefill::fused_experts_prefill;
}  // namespace ttnn
