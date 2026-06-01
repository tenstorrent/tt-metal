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
    // Number of routing-selected experts to actually run (the count of nonzero routing-weight
    // columns). Must be in [1, gate_up_weights.size()]. The selected experts' down outputs are
    // scaled by their routing weights and summed into the single [1, 1, H] output row.
    uint32_t num_experts{};

    // SwiGLU intermediate size (I). gate_up weights are [H, 2I], down weights are [I, H].
    uint32_t intermediate_size{};

    // Clamp limit applied inside the SwiGLU activation: silu(clamp(gate, max=limit)) * clamp(up, -limit, limit).
    float swiglu_limit{};

    tt::tt_metal::MemoryConfig output_memory_config{};
};

// All tensors flowing in/out of the operation. This op is the concrete example of an op that takes
// an *array* of tensors: one gate_up / down weight tensor per expert.
//
// Decode-only: sequence length T == 1, so activations are effectively [1, 1, 1, H].
//
// Expert selection/scaling is fully on-device: the i-th weight pair is scaled by column i of the
// on-device `routing_weights` tensor (no host-side `expert_ids` / "hit" list). Experts whose routing
// weight is zero contribute nothing.
struct tensor_args_t {
    // Activations, [1, 1, 1, H] (decode, T == 1).
    const Tensor& input_tensor;

    // Per-token routing weights, [1, 1, 1, E], where E == gate_up_weights.size(). Column i scales the
    // i-th expert's output.
    const Tensor& routing_weights;

    // One gate_up weight tensor per expert, each [H, 2I] (matmul-ready / transposed).
    std::vector<Tensor> gate_up_weights;

    // One down weight tensor per expert, each [I, H] (matmul-ready / transposed).
    std::vector<Tensor> down_weights;
};

using spec_return_value_t = TensorSpec;

using tensor_return_value_t = Tensor;

}  // namespace ttnn::operations::experimental::deepseek::moe::fused_experts
