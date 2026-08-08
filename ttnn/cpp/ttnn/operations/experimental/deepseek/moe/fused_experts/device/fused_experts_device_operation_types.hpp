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

    // Router top-k width: how many expert ids each token row of `routing_indices` carries. The op
    // reads exactly this many ids per token and normalizes over exactly this many scores, so it is
    // the router's k, not a bound.
    uint32_t top_k{};

    // The per-token routing weights are the selected scores renormalized to sum to 1 and scaled:
    //   w[b, i] = routed_scaling_factor * s[b, i] / (sum_j s[b, j] + routing_eps),
    // matching the reference's normalize-then-scale tail.
    float routed_scaling_factor{};
    float routing_eps{};

    tt::tt_metal::MemoryConfig output_memory_config{};
};

// All tensors flowing in/out of the operation. This op is the concrete example of an op that takes
// an *array* of tensors: one gate_up / down weight tensor per expert.
//
// B token rows are computed together, with B <= 32 so they occupy a single tile row: activations are
// [1, 1, B, H].
//
// Expert selection/scaling is fully on-device (no host-side `expert_ids` / "hit" list) and stays in
// the sparse form the router produces it: each token row carries its k selected expert ids, and the
// op reads those experts' scores out of the E-wide score row, normalizes them per token and scales
// them itself. That is the same information the op consumes internally -- the deduplicated id list
// and per-token weights it publishes in cb_bcast -- so nothing has to scatter k values out to E
// columns that would then be scanned straight back down to k.
struct tensor_args_t {
    // Activations, [1, 1, B, H] with B <= 32 token rows.
    const Tensor& input_tensor;

    // Selected expert ids, [1, 1, B, top_k] TILE, in their native tile layout: either UINT16 (the
    // index output of `ttnn.topk`, consumed unmodified) or BFLOAT16 (a `ttnn.embedding` gather from
    // a frozen id table, which is the only dtype that op gathers; exact for E <= 256).
    Tensor routing_indices;

    // Per-expert scores, [1, 1, B, E] TILE bfloat16 -- the UNBIASED router scores. The op gathers
    // s[b, routing_indices[b, j]] from these, so it must be the score tensor the ids index into
    // (the selection may have ranked by a bias-corrected copy of it).
    Tensor routing_scores;

    // One gate_up weight tensor per expert, each [H, 2I] (matmul-ready / transposed).
    std::vector<Tensor> gate_up_weights;

    // One down weight tensor per expert, each [I, H] (matmul-ready / transposed).
    std::vector<Tensor> down_weights;
};

using spec_return_value_t = tt::tt_metal::TensorSpec;

using tensor_return_value_t = Tensor;

}  // namespace ttnn::operations::experimental::deepseek::moe::fused_experts
