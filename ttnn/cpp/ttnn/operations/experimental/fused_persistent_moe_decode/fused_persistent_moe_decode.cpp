// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "fused_persistent_moe_decode.hpp"

namespace ttnn::operations::experimental::fused_persistent_moe_decode {

ttnn::Tensor FusedPersistentMoeDecodeOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& topk_expert_indices,
    const ttnn::Tensor& topk_expert_weights,
    const ttnn::Tensor& w1_experts,
    const ttnn::Tensor& w3_experts,
    const ttnn::Tensor& w2_experts) {
    return ttnn::device_operation::launch<ExecuteFusedPersistentMoeDecodeDeviceOperation>(
        {input_tensor.memory_config()},
        {input_tensor, topk_expert_indices, topk_expert_weights, w1_experts, w3_experts, w2_experts});
}

} // namespace ttnn::operations::experimental::fused_persistent_moe_decode
