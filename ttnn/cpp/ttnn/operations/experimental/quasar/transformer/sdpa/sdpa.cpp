// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <utility>

#include "ttnn/operations/experimental/quasar/transformer/sdpa/sdpa.hpp"

#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/experimental/quasar/transformer/sdpa/device/sdpa_device_operation.hpp"
#include "ttnn/operations/experimental/quasar/transformer/sdpa/device/joint_sdpa_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/device.hpp"

namespace ttnn::operations::experimental::quasar::transformer {

ttnn::Tensor scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const std::optional<ttnn::Tensor>& attn_mask,
    bool is_causal,
    std::optional<float> scale,
    std::optional<uint32_t> sliding_window_size,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<ttnn::Tensor>& attention_sink,
    const std::optional<ttnn::Tensor>& cu_window_seqlens,
    uint32_t windowed_q_token_offset,
    const std::optional<ttnn::Tensor>& windowed_q_token_offset_tensor) {
    [[maybe_unused]] auto arch = input_tensor_q.storage_type() == StorageType::DEVICE
                                     ? input_tensor_q.device()->arch()
                                     : ttnn::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi2, true, false, false);

    // PyTorch semantics: softmax(Q·Kᵀ * scale + mask) · V, where `scale` applies
    // to Q·Kᵀ only and the mask is added unscaled.
    //
    // The compute kernel folds `scale` into the softmax exponent as a
    // performance optimization:
    //     exp((QK + mask - row_max) * scale)
    //   = exp(QK*scale + mask*scale - row_max*scale)
    // which scales the mask along with QK, diverging from PyTorch semantics.
    //
    // Pre-multiply the mask by 1/scale so the kernel's subsequent *scale
    // restores the original mask magnitude inside softmax. QK remains scaled
    // exactly once.
    //
    // Windowed mode synthesizes a {0, -inf} block-diagonal mask on-device from cu_window_seqlens;
    // pre-scaling is unnecessary (0/-inf are scale-invariant), so attn_mask is left empty.
    std::optional<ttnn::Tensor> effective_mask = attn_mask;
    if (attn_mask.has_value()) {
        const float effective_scale =
            scale.value_or(1.0f / std::sqrt(static_cast<float>(input_tensor_q.padded_shape()[-1])));
        if (effective_scale != 1.0f) {
            effective_mask = ttnn::multiply(attn_mask.value(), 1.0f / effective_scale);
        }
    }

    return ttnn::prim::qsr::sdpa(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        effective_mask,
        std::nullopt,  // page_table
        attention_sink,
        is_causal,
        scale,
        sliding_window_size,
        std::nullopt,  // chunk_start_idx
        std::nullopt,  // chunk_start_idx_tensor
        false,  // use_mla
        std::nullopt,  // head_dim_v
        memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        std::move(program_config),
        kernel_config_val,
        cu_window_seqlens,
        windowed_q_token_offset,
        windowed_q_token_offset_tensor);
}

// Legacy: chunk_start_idx as scalar (part of program cache key).
ttnn::Tensor chunked_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& page_table_tensor,
    int64_t chunk_start_idx,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<ttnn::operations::transformer::PagedCacheGeometryOverride> paged_cache_geometry) {
    [[maybe_unused]] auto arch = input_tensor_q.storage_type() == StorageType::DEVICE
                                     ? input_tensor_q.device()->arch()
                                     : ttnn::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi2, true, false, false);

    return ttnn::prim::qsr::sdpa(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        std::nullopt,        // attn_mask
        page_table_tensor,   // page_table
        std::nullopt,        // attention_sink
        /*is_causal=*/true,  // Always causal for chunked version
        scale,
        std::nullopt,  // sliding_window_size (not supported yet)
        chunk_start_idx,
        std::nullopt,  // chunk_start_idx_tensor
        false,  // use_mla
        std::nullopt,  // head_dim_v
        memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        std::move(program_config),
        kernel_config_val,
        std::nullopt,  // cu_window_seqlens
        0,  // windowed_q_token_offset (windowed mode only)
        std::nullopt,  // windowed_q_token_offset_tensor
        paged_cache_geometry);
}

// Flexible: chunk_start_idx in device tensor [1]; read at runtime (for tracing).
ttnn::Tensor chunked_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& page_table_tensor,
    const ttnn::Tensor& chunk_start_idx_tensor,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<ttnn::operations::transformer::PagedCacheGeometryOverride> paged_cache_geometry) {
    [[maybe_unused]] auto arch = input_tensor_q.storage_type() == StorageType::DEVICE
                                     ? input_tensor_q.device()->arch()
                                     : ttnn::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi2, true, false, false);

    return ttnn::prim::qsr::sdpa(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        std::nullopt,        // attn_mask
        page_table_tensor,   // page_table
        std::nullopt,  // attention_sink
        /*is_causal=*/true,
        scale,
        std::nullopt,  // sliding_window_size
        std::nullopt,
        chunk_start_idx_tensor,
        false,  // use_mla
        std::nullopt,  // head_dim_v
        memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        std::move(program_config),
        kernel_config_val,
        std::nullopt,  // cu_window_seqlens
        0,  // windowed_q_token_offset (windowed mode only)
        std::nullopt,  // windowed_q_token_offset_tensor
        paged_cache_geometry);
}

std::tuple<ttnn::Tensor, ttnn::Tensor> joint_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& joint_tensor_q,
    const ttnn::Tensor& joint_tensor_k,
    const ttnn::Tensor& joint_tensor_v,
    const std::string& joint_strategy,
    ttnn::operations::transformer::SDPAProgramConfig program_config,
    std::optional<float> scale,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    auto output_tensors = ttnn::prim::qsr::joint_scaled_dot_product_attention(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        joint_tensor_q,
        joint_tensor_k,
        joint_tensor_v,
        joint_strategy,
        program_config,
        scale,
        compute_kernel_config);
    return {output_tensors[prim::qsr::JOINT_SDPA_OUTPUT_IDX],
            output_tensors[prim::qsr::JOINT_SDPA_JOINT_OUTPUT_IDX]};
}

ttnn::Tensor flash_mla_prefill(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const uint32_t head_dim_v,
    const std::optional<ttnn::Tensor>& input_tensor_v,
    const std::optional<ttnn::Tensor>& attn_mask,
    bool is_causal,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    [[maybe_unused]] auto arch = input_tensor_q.storage_type() == StorageType::DEVICE
                                     ? input_tensor_q.device()->arch()
                                     : ttnn::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi2, true, false, false);

    return ttnn::prim::qsr::sdpa(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        attn_mask,
        std::nullopt,  // page_table
        std::nullopt,  // attention_sink
        is_causal,
        scale,
        std::nullopt,  // sliding_window_size (not supported yet)
        std::nullopt,  // chunk_start_idx
        std::nullopt,  // chunk_start_idx_tensor
        true,  // use_mla
        head_dim_v,
        memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        std::move(program_config),
        kernel_config_val);
}

ttnn::Tensor chunked_flash_mla_prefill(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const uint32_t head_dim_v,
    const ttnn::Tensor& page_table_tensor,
    int64_t chunk_start_idx,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    [[maybe_unused]] auto arch = input_tensor_q.storage_type() == StorageType::DEVICE
                                     ? input_tensor_q.device()->arch()
                                     : ttnn::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi2, true, false, false);

    return ttnn::prim::qsr::sdpa(
        input_tensor_q,
        input_tensor_k,
        std::nullopt,        // V is implied by K in MLA mode
        std::nullopt,        // attn_mask
        page_table_tensor,   // page_table
        std::nullopt,  // attention_sink
        /*is_causal=*/true,
        scale,
        std::nullopt,  // sliding_window_size (not supported yet)
        chunk_start_idx,
        std::nullopt,  // chunk_start_idx_tensor
        true,  // use_mla
        head_dim_v,
        memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        std::move(program_config),
        kernel_config_val);
}

}  // namespace ttnn::operations::experimental::quasar::transformer
