// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"

namespace ttnn {
namespace operations::transformer {

}  // namespace operations::transformer

namespace transformer {

ttnn::Tensor scaled_dot_product_attention_decode(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    bool is_causal = true,
    const std::optional<const Tensor>& attn_mask = std::nullopt,
    const std::vector<uint32_t>& cur_pos = std::vector<uint32_t>(),
    const std::optional<const Tensor>& cur_pos_tensor = std::nullopt,
    const std::optional<const Tensor>& attention_sink = std::nullopt,
    std::optional<float> scale = std::nullopt,
    std::optional<uint32_t> sliding_window_size = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<SDPAProgramConfig> program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

ttnn::Tensor paged_scaled_dot_product_attention_decode(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& page_table_tensor,
    bool is_causal = true,
    const std::optional<const Tensor>& attn_mask = std::nullopt,
    const std::optional<const Tensor>& cur_pos_tensor = std::nullopt,
    const std::optional<const Tensor>& attention_sink = std::nullopt,
    std::optional<float> scale = std::nullopt,
    std::optional<uint32_t> sliding_window_size = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<SDPAProgramConfig> program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

ttnn::Tensor flash_multi_latent_attention_decode(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const std::optional<const Tensor>& input_tensor_v,
    uint32_t head_dim_v,
    bool is_causal = true,
    const std::optional<const Tensor>& attn_mask = std::nullopt,
    const std::vector<uint32_t>& cur_pos = std::vector<uint32_t>(),
    const std::optional<const Tensor>& cur_pos_tensor = std::nullopt,
    const std::optional<const Tensor>& attention_sink = std::nullopt,
    std::optional<float> scale = std::nullopt,
    std::optional<uint32_t> sliding_window_size = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<SDPAProgramConfig> program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

ttnn::Tensor paged_flash_multi_latent_attention_decode(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const std::optional<const Tensor>& input_tensor_v,
    uint32_t head_dim_v,
    const ttnn::Tensor& page_table_tensor,
    bool is_causal = true,
    const std::optional<const Tensor>& attn_mask = std::nullopt,
    const std::optional<const Tensor>& cur_pos_tensor = std::nullopt,
    const std::optional<const Tensor>& attention_sink = std::nullopt,
    std::optional<float> scale = std::nullopt,
    std::optional<uint32_t> sliding_window_size = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<SDPAProgramConfig> program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace transformer

}  // namespace ttnn
