// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/types.hpp"

namespace ttnn::transformer {

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
    std::optional<operations::transformer::SDPAProgramConfig> program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    std::optional<bool> share_cache = std::nullopt);

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
    std::optional<operations::transformer::SDPAProgramConfig> program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    std::optional<operations::transformer::PagedCacheGeometryOverride> paged_cache_geometry = std::nullopt,
    std::optional<uint32_t> cache_position_modulo = std::nullopt,
    uint32_t spec_multi_pos_tiles = 0);

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
    std::optional<operations::transformer::SDPAProgramConfig> program_config = std::nullopt,
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
    std::optional<operations::transformer::SDPAProgramConfig> program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

// paged_flash_multi_latent_attention_decode that also returns the log-sum-exp of every (user, Q head):
// (output, lse), lse fp32 [1, B, nh, 1] TILE (each head's value in column 0 of its tile row, padding 0), in the
// kernel's logit space: lse = log(sum_k exp(scale * q.k)). Users with cur_pos = -1 get output rows 0 and
// lse = -1e30. Needs one KV head, no attention sink and a TILE layout Q.
std::tuple<ttnn::Tensor, ttnn::Tensor> paged_flash_multi_latent_attention_decode_with_lse(
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
    std::optional<operations::transformer::SDPAProgramConfig> program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn::transformer
