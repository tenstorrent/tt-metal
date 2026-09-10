// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar::transformer {

ttnn::Tensor scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const std::optional<ttnn::Tensor>& attn_mask = std::nullopt,
    bool is_causal = true,
    std::optional<float> scale = std::nullopt,
    std::optional<uint32_t> sliding_window_size = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<operations::transformer::SDPAProgramConfig> program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<ttnn::Tensor>& attention_sink = std::nullopt,
    const std::optional<ttnn::Tensor>& cu_window_seqlens = std::nullopt,
    /// Windowed mode only. Global row index of Q row 0, for a Q holding a contiguous slice of a longer
    /// sequence: Q and the output are indexed locally while cu_window_seqlens and K/V stay global, so this
    /// locates the slice among the windows. Must be a multiple of TILE_HEIGHT and satisfy offset+Sq <= Sk.
    uint32_t windowed_q_token_offset = 0,
    /// Windowed mode only. Per-device form of the offset above: a 1-element int32/uint32 ROW_MAJOR device
    /// tensor, read at runtime rather than baked into the program. Shard it on the sequence-parallel axis
    /// so every device runs the SAME program yet sees its own origin. Overrides the scalar when set.
    const std::optional<ttnn::Tensor>& windowed_q_token_offset_tensor = std::nullopt);

/// Chunked SDPA over paged K/V: one Q chunk per call, K/V in paged layout.
/// Two overloads: legacy (chunk_start_idx as int) or flexible (chunk_start_idx_tensor on device).
///
/// Legacy: chunk start index as scalar.
ttnn::Tensor chunked_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& page_table_tensor,
    int64_t chunk_start_idx,  // Must be a multiple of program_config.q_chunk_size
    std::optional<float> scale = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<operations::transformer::SDPAProgramConfig> program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    // Geometry override for an HMA-shared paged cache. Q drives head_dim; supply this
    // call's view (block_size + num_kv_heads) when the cache was allocated for a different
    // layer. nullopt ⇒ cache shape.
    std::optional<operations::transformer::PagedCacheGeometryOverride> paged_cache_geometry = std::nullopt);

/// Flexible: chunk start index in device tensor [1] (int32). Read at runtime; use for trace.
ttnn::Tensor chunked_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& page_table_tensor,
    const ttnn::Tensor& chunk_start_idx_tensor,
    std::optional<float> scale = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<operations::transformer::SDPAProgramConfig> program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    std::optional<operations::transformer::PagedCacheGeometryOverride> paged_cache_geometry = std::nullopt);

std::tuple<ttnn::Tensor, ttnn::Tensor> joint_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& joint_tensor_q,
    const ttnn::Tensor& joint_tensor_k,
    const ttnn::Tensor& joint_tensor_v,
    const std::string& joint_strategy,
    operations::transformer::SDPAProgramConfig program_config,
    std::optional<float> scale = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

ttnn::Tensor flash_mla_prefill(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    uint32_t head_dim_v,
    const std::optional<ttnn::Tensor>& input_tensor_v = std::nullopt,
    const std::optional<ttnn::Tensor>& attn_mask = std::nullopt,
    bool is_causal = true,
    std::optional<float> scale = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<operations::transformer::SDPAProgramConfig> program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

ttnn::Tensor chunked_flash_mla_prefill(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    uint32_t head_dim_v,
    const ttnn::Tensor& page_table_tensor,
    int64_t chunk_start_idx,
    std::optional<float> scale = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<operations::transformer::SDPAProgramConfig> program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn::operations::experimental::quasar::transformer
