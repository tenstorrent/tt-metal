// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "cpp/ttnn/global_semaphore.hpp"

namespace ttnn {
namespace operations::experimental::speculative_execution {

using SDPAProgramConfig = ttnn::operations::transformer::SDPAProgramConfig;

struct ExecuteSpeculativeScaledDotProductAttentionDecode {
    static std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor_q,
        const ttnn::Tensor& input_tensor_k,
        const ttnn::Tensor& input_tensor_v,
        std::optional<float> lambda_,
        const bool is_causal = true,
        const std::optional<const Tensor>& attn_mask = std::nullopt,
        const std::vector<uint32_t>& cur_pos = std::vector<uint32_t>(),
        const std::optional<const Tensor>& cur_pos_tensor = std::nullopt,
        std::optional<float> scale = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<SDPAProgramConfig> program_config = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const std::optional<Tensor>& priority_tensor = std::nullopt,
        const std::optional<Tensor>& other_priority_tensor = std::nullopt,
        const bool ccl_enabled = false,
        const std::optional<global_semaphore::MultiDeviceGlobalSemaphore>& multi_device_global_semaphore =
            std::nullopt);

    static std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> invoke(
        const ttnn::Tensor& input_tensor_q,
        const ttnn::Tensor& input_tensor_k,
        const ttnn::Tensor& input_tensor_v,
        std::optional<float> lambda_,
        const bool is_causal = true,
        const std::optional<const Tensor>& attn_mask = std::nullopt,
        const std::vector<uint32_t>& cur_pos = std::vector<uint32_t>(),
        const std::optional<const Tensor>& cur_pos_tensor = std::nullopt,
        std::optional<float> scale = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<SDPAProgramConfig> program_config = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const std::optional<Tensor>& priority_tensor = std::nullopt,
        const std::optional<Tensor>& other_priority_tensor = std::nullopt,
        const bool ccl_enabled = false,
        const std::optional<global_semaphore::MultiDeviceGlobalSemaphore>& multi_device_global_semaphore =
            std::nullopt);
};

}  // namespace operations::experimental::speculative_execution

namespace experimental {

constexpr auto speculative_scaled_dot_product_attention_decode = ttnn::register_operation<
    "ttnn::experimental::speculative_scaled_dot_product_attention_decode",
    ttnn::operations::experimental::speculative_execution::ExecuteSpeculativeScaledDotProductAttentionDecode>();

}  // namespace experimental

}  // namespace ttnn
