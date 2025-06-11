// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"

namespace ttnn {
namespace operations::transformer {

struct ExecuteScaledDotProductAttention {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor_q,
        const ttnn::Tensor& input_tensor_k,
        const ttnn::Tensor& input_tensor_v,
        const std::optional<ttnn::Tensor>& attn_mask = std::nullopt,
        bool is_causal = true,
        std::optional<float> scale = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<SDPAProgramConfig> program_config = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor_q,
        const ttnn::Tensor& input_tensor_k,
        const ttnn::Tensor& input_tensor_v,
        const std::optional<ttnn::Tensor>& attn_mask = std::nullopt,
        bool is_causal = true,
        std::optional<float> scale = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<SDPAProgramConfig> program_config = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

struct ExecuteChunkedScaledDotProductAttention {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor_q,
        const ttnn::Tensor& input_tensor_k,
        const ttnn::Tensor& input_tensor_v,
        const ttnn::Tensor& page_table_tensor,
        int64_t chunk_start_idx,
        std::optional<float> scale = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<SDPAProgramConfig> program_config = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor_q,
        const ttnn::Tensor& input_tensor_k,
        const ttnn::Tensor& input_tensor_v,
        const ttnn::Tensor& page_table_tensor,
        int64_t chunk_start_idx,
        std::optional<float> scale = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<SDPAProgramConfig> program_config = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

struct ExecuteJointAttention {
    static std::tuple<ttnn::Tensor, ttnn::Tensor> invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor_q,
        const ttnn::Tensor& input_tensor_k,
        const ttnn::Tensor& input_tensor_v,
        const ttnn::Tensor& joint_tensor_q,
        const ttnn::Tensor& joint_tensor_k,
        const ttnn::Tensor& joint_tensor_v,
        const std::string& joint_strategy,
        SDPAProgramConfig program_config,
        std::optional<float> scale = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

    static std::tuple<ttnn::Tensor, ttnn::Tensor> invoke(
        const ttnn::Tensor& input_tensor_q,
        const ttnn::Tensor& input_tensor_k,
        const ttnn::Tensor& input_tensor_v,
        const ttnn::Tensor& joint_tensor_q,
        const ttnn::Tensor& joint_tensor_k,
        const ttnn::Tensor& joint_tensor_v,
        const std::string& joint_strategy,
        SDPAProgramConfig program_config,
        std::optional<float> scale = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

}  // namespace operations::transformer

namespace transformer {

constexpr auto scaled_dot_product_attention = ttnn::register_operation<
    "ttnn::transformer::scaled_dot_product_attention",
    ttnn::operations::transformer::ExecuteScaledDotProductAttention>();

constexpr auto chunked_scaled_dot_product_attention = ttnn::register_operation<
    "ttnn::transformer::chunked_scaled_dot_product_attention",
    ttnn::operations::transformer::ExecuteChunkedScaledDotProductAttention>();

constexpr auto joint_scaled_dot_product_attention = ttnn::register_operation<
    "ttnn::transformer::joint_scaled_dot_product_attention",
    ttnn::operations::transformer::ExecuteJointAttention>();

}  // namespace transformer

}  // namespace ttnn
