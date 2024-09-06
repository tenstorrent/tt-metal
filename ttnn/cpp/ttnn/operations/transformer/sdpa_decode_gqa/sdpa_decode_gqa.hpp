// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"

namespace ttnn {
namespace operations::transformer {

struct ExecuteScaledDotProductAttentionGQADecode {
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor &input_tensor_q,
        const ttnn::Tensor &input_tensor_k,
        const ttnn::Tensor &input_tensor_v,
        const std::vector<uint32_t> cur_pos,
        std::optional<bool> transpose_q,
        std::optional<bool> share_cache,
        std::optional<float> scale = std::nullopt,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<SDPAProgramConfig> program_config = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

    static ttnn::Tensor invoke(
        const ttnn::Tensor &input_tensor_q,
        const ttnn::Tensor &input_tensor_k,
        const ttnn::Tensor &input_tensor_v,
        const std::vector<uint32_t> cur_pos,
        std::optional<bool> transpose_q,
        std::optional<bool> share_cache,
        std::optional<float> scale = std::nullopt,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<SDPAProgramConfig> program_config = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

}  // namespace operations::transformer

namespace transformer {

constexpr auto scaled_dot_product_attention_decode_gqa = ttnn::register_operation_with_auto_launch_op<
    "ttnn::transformer::scaled_dot_product_attention_decode_gqa",
    ttnn::operations::transformer::ExecuteScaledDotProductAttentionGQADecode>();

}  // namespace transformer

}  // namespace ttnn
