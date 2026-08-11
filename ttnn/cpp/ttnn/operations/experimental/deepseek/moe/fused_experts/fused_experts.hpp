// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::deepseek::moe {

// Fused routed-expert FFN for DeepSeek V4-Flash, for a batch of B <= 32 tokens.
//
// Replaces the per-expert host loop
//   for e in experts:
//       gate_up = matmul(x, gate_up_w[e]); act = swiglu(gate_up, intermediate, limit)
//       down    = matmul(act, down_w[e]);  acc += down * w[:, e]
// with a single device operation. Expert selection/scaling is derived on-device from the router's
// output (no host-side expert-id / "hit" list) -- see ROUTING INPUT below.
//
// BATCHING. The B tokens are the rows of dim -2 and share a single 32-row tile, so a [1, B, S, H]
// activation must be folded into [1, 1, B*S, H] by the caller. Batching costs essentially nothing
// and saves the dominant expense of the op: the expert ids are the DEDUPLICATED UNION of the tokens'
// selections, so an expert several tokens picked has its weights fetched from DRAM exactly once and
// its matmuls run exactly once (the tile row covers all B tokens at once), with the tokens separated
// only at the final accumulation by their own routing weights. Weight traffic is therefore set by
// the number of DISTINCT experts, not by the token count -- which is the whole point, since this op
// is DRAM-bound on the weight fetch. Tokens are capped at one tile row because the resident
// activation and the gathered activation block (the dominant L1 consumer) would otherwise scale with
// the number of tile rows.
//
// EXPERT BLOCKING. `num_experts` is not bounded by L1: `experts_block_size` sets how many experts'
// activations are gathered and held at once, and the op runs the selected experts in blocks of that
// size. This is what lets a batch whose tokens select DISJOINT experts run at all -- 32 tokens at
// top_k 6 select up to 192 distinct experts, whose activation block would be ~13 MB per core if held
// all at once. Blocking changes no arithmetic and no DRAM traffic (each expert is still fetched
// exactly once); it costs one gather/broadcast synchronization per block instead of one for the whole
// op, and it double-buffers the activation block so consecutive blocks pipeline.
//
// ROUTING INPUT. The routing decision stays in the sparse form the router produces it: each token's
// selected expert ids plus the unbiased score row they index, both passed through in their native
// TILE layout. The op gathers each token's k scores, normalizes them (sum to 1, then
// `routed_scaling_factor`) and derives the hit ids and per-token weights itself. That is what the
// op's internals already use, so widening the k values into an E-wide weight row would be a
// temporary built by a scatter + normalize + relayout chain purely for the first kernel to scan it
// straight back down to k -- and it turns an O(E x B) hit scan into an O(B x k) one.
//
// Args:
//   input_tensor:     activations, [1, 1, B, H] with B <= 32 token rows.
//   routing_indices:  selected expert ids, [1, 1, B, top_k] TILE, either UINT16 (a `ttnn.topk`
//                     index output) or BFLOAT16 (a `ttnn.embedding` gather from an id table; exact
//                     for E <= 256, and the only dtype that op gathers).
//   routing_scores:   unbiased per-expert scores, [1, 1, B, E] TILE bfloat16 -- the tensor the ids
//                     index into (the selection may have ranked a bias-corrected copy of it).
//   top_k:            ids per token (<= 16); 0 takes it from routing_indices.
//   routed_scaling_factor / routing_eps: the normalize tail applied per token,
//                     w = scale * s / (sum(s) + eps).
//   gate_up_weights:  one [H, 2I] weight tensor per expert (all experts provided), with the
//                     gate/up columns interleaved at tile (32-col) granularity so each core's
//                     [H, 64] DRAM shard is the [gate_tile | up_tile] pair for its output tile.
//   down_weights:     one [I, H] weight tensor per expert.
//   num_experts:      number of routing-selected ("hit") experts to run: the size of the union of
//                     the tokens' selections. May be passed as an upper bound -- useful when B > 1,
//                     where the exact union size is data dependent while the compiled program is
//                     not -- at the cost of one redundant weight fetch per unused slot; the surplus
//                     experts contribute nothing.
//   intermediate_size: SwiGLU intermediate size I.
//   swiglu_limit:     clamp limit used by the SwiGLU activation.
//   experts_block_size: experts to hold in L1 at once; 0 (the default) means all `num_experts`,
//                     reproducing the single-block pipeline exactly. Any smaller value trades one
//                     extra chip-wide synchronization per block for an L1 footprint set by the block
//                     rather than by `num_experts`. Since blocking double-buffers the activation
//                     block, the largest usable block is about half the largest usable single block.
//   memory_config:    optional output memory config (defaults to the input's).
//
// Returns a [1, B, H] BFLOAT16 TILE tensor (the B token rows padded to a 32-row tile):
//   act       = silu(clamp(gate, max=limit)) * clamp(up, -limit, limit),
//               where [gate, up] = x @ gate_up_w[hit_ids[i]];
//   output[b] = sum_i w[b, hit_ids[i]] * (act[b] @ down_w[hit_ids[i]]),
// with hit_ids the routing-selected experts in ascending order and w the normalized weights above
// (zero for a token that did not select the expert). The I SwiGLU columns are distributed across the
// 8x8 compute grid, each core reading its [H, 128] interleaved gate/up shard and its [I, H/64] down
// shard per selected expert in a single NoC read from the DRAM ND-sharded weights; the SwiGLU
// activation is gathered onto core {0,0} and broadcast back for the down matmul. All three input
// tensors are TILE layout.
Tensor fused_experts(
    const Tensor& input_tensor,
    const Tensor& routing_indices,
    const Tensor& routing_scores,
    const std::vector<Tensor>& gate_up_weights,
    const std::vector<Tensor>& down_weights,
    uint32_t num_experts,
    uint32_t intermediate_size,
    float swiglu_limit,
    uint32_t top_k = 0,
    float routed_scaling_factor = 1.0F,
    float routing_eps = 0.0F,
    uint32_t experts_block_size = 0,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental::deepseek::moe
