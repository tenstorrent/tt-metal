// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe.hpp"
#include "device/moe_device_operation.hpp"

namespace ttnn::operations::experimental::moe {

ttnn::Tensor ExecuteMoE::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& w0_w1_tensor,
    const ttnn::Tensor& w2_tensor,
    const ttnn::Tensor& output_tensor,
    const uint32_t num_experts,
    const uint32_t layer_id,
    const uint32_t num_tokens_total,
    const uint32_t output_height_shard_dim,
    const uint32_t output_width_shard_dim) {
    return ttnn::prim::moe(
        input_tensor,
        w0_w1_tensor,
        w2_tensor,
        output_tensor,
        num_experts,
        layer_id,
        num_tokens_total,
        output_height_shard_dim,
        output_width_shard_dim);
}

}  // namespace ttnn::operations::experimental::moe
