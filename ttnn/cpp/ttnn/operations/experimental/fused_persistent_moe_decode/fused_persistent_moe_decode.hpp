// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/experimental/fused_persistent_moe_decode/device/fused_persistent_moe_decode_device_operation.hpp"

namespace ttnn::operations::experimental::fused_persistent_moe_decode {

struct FusedPersistentMoeDecodeOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& topk_expert_indices,
        const ttnn::Tensor& topk_expert_weights,
        const ttnn::Tensor& w1_experts,
        const ttnn::Tensor& w3_experts,
        const ttnn::Tensor& w2_experts);
};

} // namespace ttnn::operations::experimental::fused_persistent_moe_decode

namespace ttnn::experimental {
    constexpr auto fused_persistent_moe_decode = ttnn::register_operation<
        "ttnn::experimental::fused_persistent_moe_decode",
        ttnn::operations::experimental::fused_persistent_moe_decode::FusedPersistentMoeDecodeOperation>();
}
