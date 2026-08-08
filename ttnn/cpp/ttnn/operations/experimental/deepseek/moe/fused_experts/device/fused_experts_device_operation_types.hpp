// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::experimental::deepseek::moe::fused_experts {

// Non-tensor parameters of the fused routed-expert FFN.
struct operation_attributes_t {
    // Number of routing-selected experts to actually run: the count of routing-weight columns that
    // are nonzero for at least one token row (the union of the rows' selections). Must be in
    // [1, gate_up_weights.size()]. Every selected expert is evaluated for every token row, its
    // down output scaled by that row's routing weight (zero where the row did not select it), and
    // summed into the [1, B, H] output.
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

    tt::tt_metal::MemoryConfig output_memory_config{};
};

// All tensors flowing in/out of the operation. This op is the concrete example of an op that takes
// an *array* of tensors: one gate_up / down weight tensor per expert.
//
// B token rows are computed together, with B <= 32 so they occupy a single tile row: activations are
// [1, 1, B, H].
//
// Expert selection/scaling is fully on-device: the i-th weight pair is scaled, for token row b, by
// element [b, i] of the on-device `routing_weights` tensor (no host-side `expert_ids` / "hit" list).
// Experts whose routing weight is zero in every row are skipped entirely; rows that did not select
// a run expert contribute nothing to it because their weight is zero.
struct tensor_args_t {
    // Activations, [1, 1, B, H] with B <= 32 token rows.
    const Tensor& input_tensor;

    // Per-token routing weights, [1, 1, B, E], where E == gate_up_weights.size(). Element [b, i]
    // scales the i-th expert's contribution to token row b.
    const Tensor& routing_weights;

    // One gate_up weight tensor per expert, each [H, 2I] (matmul-ready / transposed).
    std::vector<Tensor> gate_up_weights;

    // One down weight tensor per expert, each [I, H] (matmul-ready / transposed).
    std::vector<Tensor> down_weights;
};

using spec_return_value_t = tt::tt_metal::TensorSpec;

using tensor_return_value_t = Tensor;

}  // namespace ttnn::operations::experimental::deepseek::moe::fused_experts
