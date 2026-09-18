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
// ROUTING INPUT. The routing decision reaches the op in one of two forms, exactly one of which must
// be supplied:
//
//   * `routing_indices` -- the router's own top-k output: each token's selected expert ids, plus the
//     score row they index. Ids are TILE; scores are TILE or (decode, B == 1) ROW_MAJOR. The op
//     gathers each token's k scores, normalizes them (sum to 1, then `routed_scaling_factor`) and
//     derives the hit ids and per-token weights itself. That is what the op's internals already use,
//     so widening the k values into an E-wide weight row would be a temporary built by a scatter +
//     normalize + relayout chain purely for the first kernel to scan it straight back down to k --
//     and it turns an O(E x B) hit scan into an O(B x k) one.
//
//   * `ranking_scores` -- the E-wide score row the router would have ranked (typically a
//     bias-corrected copy of `routing_scores`), with `top_k` given explicitly. The op finds each
//     token's top-k largest entries itself, inside the leader kernel that already reads the whole
//     score row, so the router does not have to launch `ttnn.topk` and round-trip its id output
//     through DRAM. The weights are still the *unbiased* `routing_scores` at the selected ids, which
//     is what makes "rank on the biased row, weight with the uncorrected one" a single op. Pass the
//     same tensor for both when they coincide (the op then reads it once).
//
// INPUT LAYOUTS. Three shapes are accepted for the activation, all consumed as one tile row of B
// tokens:
//   * TILE, [1, 1, B, H] with B <= 32 -- the prefill / batched form.
//   * ROW_MAJOR interleaved, [1, 1, 1, H] (decode) -- physically a run of 1x32 faces, so it is
//     loaded as 1x32 compute tiles with no tilize. The row is read from DRAM once by the {1,0}
//     sender and multicast to every core.
//   * ROW_MAJOR HEIGHT_SHARDED replicated in L1, [1, 1, 1 * num_cores, H] with shard [1, H] -- the
//     replica `ttnn.experimental.deepseek.all_gather_for_matmul` multicasts onto every matmul core,
//     and the layout `matmul_decode` also consumes in place. Each core already holds the full row,
//     so nothing is read from DRAM and nothing is broadcast: cb_input is aliased over the local
//     shard and merely published. The row count therefore comes from the shard height (`B`), not
//     from dim -2, which carries `num_cores * B`. This is the cheapest decode form, and it
//     removes the input broadcast barrier from the critical path.
//
// Args:
//   input_tensor:     activations, [1, 1, B, H] with B <= 32 token rows. TILE, ROW_MAJOR
//                     (B == 1; loaded as 1x32 compute tiles, no tilize), or ROW_MAJOR
//                     HEIGHT_SHARDED L1 replicated on the compute grid (see INPUT LAYOUTS).
//   routing_indices:  selected expert ids, [1, 1, B, top_k] TILE, either UINT16 (a `ttnn.topk`
//                     index output) or BFLOAT16 (a `ttnn.embedding` gather from an id table; exact
//                     for E <= 256, and the only dtype that op gathers). Mutually exclusive with
//                     `ranking_scores`.
//   routing_scores:   unbiased per-expert scores, [1, 1, B, E] bfloat16 -- TILE, or ROW_MAJOR
//                     when B == 1 (decode; LinearDecode stick, indexed linearly, no tilize).
//                     The tensor the selection indexes into -- the ranking may have used a
//                     bias-corrected copy of it, but these are the values that become weights.
//   ranking_scores:   scores to rank on, [1, 1, B, E] bfloat16, same shape/dtype rules as
//                     `routing_scores`. When supplied the op selects each token's `top_k` largest
//                     entries on device (no `ttnn.topk`, no id tensor) and they become the hit ids.
//                     Mutually exclusive with `routing_indices`; pass the same tensor as
//                     `routing_scores` when ranking and weighting use one row.
//   top_k:            ids per token (<= 16 for `routing_indices`; <= 32 for `ranking_scores`); 0
//                     takes it from `routing_indices`, and is invalid with `ranking_scores`.
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
//   memory_config:    optional output memory config. Defaults to the input's, except for a
//                     replicated input, whose L1 HEIGHT_SHARDED replica has no meaningful output
//                     layout: there the default is DRAM interleaved.
//
// Returns a [1, B, H] BFLOAT16 tensor in the input's layout (TILE, or ROW_MAJOR when B == 1):
//   act       = silu(clamp(gate, max=limit)) * clamp(up, -limit, limit),
//               where [gate, up] = x @ gate_up_w[hit_ids[i]];
//   output[b] = sum_i w[b, hit_ids[i]] * (act[b] @ down_w[hit_ids[i]]),
// with hit_ids the routing-selected experts in ascending order and w the normalized weights above
// (zero for a token that did not select the expert). Down weights stay 64-way ND-sharded along H.
// Gate_up is one [gate_32 | up_32] DRAM shard per I-tile, so TP's smaller local I yields fewer
// shards and each of the 16 cores in a group still owns at least one. With 6 selected experts and
// a 12x8 (or larger) compute grid the op uses 96 cores, 16 cores per expert (2 columns x 8 rows);
// each core covers its I-shards of that expert plus 4 of the 64 H-shards, gathers SwiGLU
// activations within the group, and the 6 groups' matching H-slices are reduced onto group 0.
// Smaller grids (or any other selected count) keep the original 8x8 grid, every core iterating
// every expert. routing_indices are TILE; routing_scores, ranking_scores and input_tensor may be
// TILE or (B==1) ROW_MAJOR.
Tensor fused_experts(
    const Tensor& input_tensor,
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
    bool two_hub_gather = true,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& routing_indices = std::nullopt,
    const std::optional<Tensor>& ranking_scores = std::nullopt);

}  // namespace ttnn::experimental::deepseek::moe
