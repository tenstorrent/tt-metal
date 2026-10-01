// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::experimental::deepseek::moe::fused_experts {

// Non-tensor parameters of the fused routed-expert FFN.
struct operation_attributes_t {
    // Number of routing-selected experts to actually run: the size of the union of the token rows'
    // selections. Must be in [1, gate_up_weights.size()]. Every selected expert is evaluated for
    // every token row, its down output scaled by that row's routing weight (zero where the row did
    // not select it), and summed into the [1, B, H] output.
    uint32_t num_experts{};

    // SwiGLU intermediate size (I). gate_up weights are [H, 2I], down weights are [I, H].
    uint32_t intermediate_size{};

    // Clamp limit applied inside the SwiGLU activation: silu(clamp(gate, max=limit)) * clamp(up, -limit, limit).
    float swiglu_limit{};

    // How many experts' SwiGLU activations are resident in L1 at once, i.e. the size of the expert
    // blocks the op processes in sequence. 0 means "all `num_experts` in one block".
    //
    // This is the knob that decouples `num_experts` from L1. The gathered activation block is the
    // dominant L1 consumer -- it lives on EVERY core -- and it is sized by this value, not by
    // `num_experts`, so a batch whose tokens select disjoint experts (up to 32 * top_k of them) runs
    // by streaming the experts through in blocks. Each expert is still fetched from DRAM exactly
    // once; the cost of a smaller block is one extra gather/broadcast synchronization per block.
    // Blocking also double-buffers that activation block so the blocks can pipeline, so the largest
    // usable value is about half what a single block allows.
    uint32_t experts_block_size{};

    // Router top-k width: how many expert ids each token row of `routing_indices` carries, or, when
    // `ranking_scores` is used, how many experts the on-device ranking selects per token. The op
    // reads/selects exactly this many ids per token and normalizes over exactly this many scores,
    // so it is the router's k, not a bound. It must be given explicitly (non-zero) whenever the
    // selection comes from `ranking_scores`, since there is no id tensor to infer it from.
    uint32_t top_k{};

    // The per-token routing weights are the selected scores renormalized to sum to 1 and scaled:
    //   w[b, i] = routed_scaling_factor * s[b, i] / (sum_j s[b, j] + routing_eps),
    // matching the reference's normalize-then-scale tail.
    float routed_scaling_factor{};
    float routing_eps{};

    // Split the per-block activation gather + broadcast across two hubs (the multicast rectangle's
    // two opposite corners), each multicasting half the I dim on its own NoC. This halves the
    // gather ingress and the broadcast egress per hub and gives the two directions a NoC each, at
    // the cost of two completion increments per block. False keeps the original single-hub pipeline
    // (one gather target, one multicast sender). Both shapes run the same kernel code; the flag only
    // selects how many hubs the host registers.
    bool two_hub_gather = true;

    tt::tt_metal::MemoryConfig output_memory_config{};
};

// All tensors flowing in/out of the operation. This op is the concrete example of an op that takes
// an *array* of tensors: one gate_up / down weight tensor per expert.
//
// B token rows are computed together, with B <= 32 so they occupy a single tile row: activations are
// [1, 1, B, H].
//
// Expert selection/scaling is fully on-device (no host-side `expert_ids` / "hit" list). The
// selection itself reaches the op in one of two forms, and exactly one of them is provided:
//
//   * `routing_indices` -- the router's own top-k output: each token row carries its k selected
//     expert ids (a `ttnn.topk` index output, or an `ttnn.embedding` gather from a frozen table).
//     The op reads and dedups them. This is the original form.
//   * `ranking_scores` -- the E-wide score row the router would have ranked (possibly a
//     bias-corrected copy). The op finds each token's top-k itself, inside the leader kernel, and
//     that becomes the selection. This removes the separate `ttnn.topk` launch and the DRAM
//     round-trip of its id output, and it is the only form that can rank on scores that never
//     leave the op.
//
// Either way the per-token weights come from `routing_scores` at the selected experts, normalized
// and scaled by the op itself -- so nothing has to scatter k values out to E columns that would
// then be scanned straight back down to k.
struct tensor_args_t {
    // Activations, [1, 1, B, H] with B <= 32 token rows. TILE, or ROW_MAJOR when B == 1
    // (decode: loaded as 1x32 compute tiles, no tilize).
    const Tensor& input_tensor;

    // Selected expert ids, [1, 1, B, top_k] TILE, in their native tile layout: either UINT16 (the
    // index output of `ttnn.topk`, consumed unmodified) or BFLOAT16 (a `ttnn.embedding` gather from
    // a frozen id table, which is the only dtype that op gathers; exact for E <= 256). Empty when
    // `ranking_scores` drives the selection instead.
    std::optional<Tensor> routing_indices;

    // Per-expert scores, [1, 1, B, E] bfloat16 -- the UNBIASED router scores. TILE, or ROW_MAJOR
    // when B == 1 (decode stick). The op gathers s[b, selected_ids[b, j]] from these, so it
    // must be the score tensor the selection was derived from (the ranking may have used a
    // bias-corrected copy of it).
    Tensor routing_scores;

    // Scores to RANK on, [1, 1, B, E] bfloat16 -- TILE, or ROW_MAJOR when B == 1. Same shape and
    // dtype rules as `routing_scores`. When present the op selects each token's `top_k` largest
    // entries itself (inside the leader kernel, which already reads the whole score row) instead
    // of reading `routing_indices`; the weights still come from `routing_scores` at those ids, so
    // "rank on the biased row, weight with the unbiased one" is one op with no host-side topk.
    // Pass the same tensor as `routing_scores` when the two coincide (the op reads it once).
    std::optional<Tensor> ranking_scores;

    // One gate_up weight tensor per expert, each [H, 2I] (matmul-ready / transposed).
    std::vector<Tensor> gate_up_weights;

    // One down weight tensor per expert, each [I, H] (matmul-ready / transposed).
    std::vector<Tensor> down_weights;
};

using spec_return_value_t = tt::tt_metal::TensorSpec;

using tensor_return_value_t = Tensor;

}  // namespace ttnn::operations::experimental::deepseek::moe::fused_experts
