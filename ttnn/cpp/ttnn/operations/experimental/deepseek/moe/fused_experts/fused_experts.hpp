// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::deepseek::moe {

// Fused routed-expert FFN for DeepSeek V4-Flash decode (sequence length T == 1).
//
// Replaces the per-expert host loop
//   for e in experts:
//       gate_up = matmul(x, gate_up_w[e]); act = swiglu(gate_up, intermediate, limit)
//       down    = matmul(act, down_w[e]);  acc += down * routing_weights[:, e]
// with a single device operation. Expert selection/scaling is read on-device from
// `routing_weights` (no host-side expert-id / "hit" list): the i-th weight pair is scaled by
// routing_weights column i, so experts with zero routing weight contribute nothing.
//
// Args:
//   input_tensor:     activations, [1, 1, 1, H] (decode).
//   routing_weights:  per-token routing weights, [1, 1, 1, E], with E == gate_up_weights.size().
//   gate_up_weights:  one [H, 2I] weight tensor per expert (all experts provided), with the
//                     gate/up columns interleaved at tile (32-col) granularity so each core's
//                     [H, 64] DRAM shard is the [gate_tile | up_tile] pair for its output tile.
//   down_weights:     one [I, H] weight tensor per expert.
//   num_experts:      number of routing-selected ("hit") experts to run; must equal the number of
//                     nonzero routing-weight columns. Also the output's leading dimension.
//   intermediate_size: SwiGLU intermediate size I (output's last dimension).
//   swiglu_limit:     clamp limit used by the SwiGLU activation.
//   memory_config:    optional output memory config (defaults to the input's).
//
// CURRENT MILESTONE: this op runs the gate_up matmul + SwiGLU activation on device for the
// routing-selected experts and returns a [num_experts, 1, I] BFLOAT16 TILE tensor (the decode token
// row padded to a 32-row tile), where
//   output[i] = silu(clamp(gate, max=limit)) * clamp(up, -limit, limit),
//   [gate, up] = x @ gate_up_w[hit_ids[i]],
// and hit_ids are the nonzero routing-weight columns in ascending order. The I output columns are
// distributed across the 8x8 compute grid, and each core reads its [H, 64] interleaved gate/up weight
// shard per selected expert in a single NoC read from DRAM ND-sharded weights. input_tensor must be
// TILE layout and routing_weights ROW_MAJOR bfloat16. Down matmul + routed accumulation are later
// milestones.
Tensor fused_experts(
    const Tensor& input_tensor,
    const Tensor& routing_weights,
    const std::vector<Tensor>& gate_up_weights,
    const std::vector<Tensor>& down_weights,
    uint32_t num_experts,
    uint32_t intermediate_size,
    float swiglu_limit,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental::deepseek::moe
