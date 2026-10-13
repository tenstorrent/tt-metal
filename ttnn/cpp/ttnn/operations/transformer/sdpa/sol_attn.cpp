// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/sol_attn.hpp"
#include "ttnn/operations/transformer/sdpa/device/sol_attn_device_operation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include <tt-metalium/hal.hpp>
#include <cmath>

namespace ttnn::transformer {

ttnn::Tensor sol_attn(
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
    const std::optional<ttnn::Tensor>& windowed_q_token_offset_tensor,
    uint32_t chunk_keep_every,
    const std::optional<ttnn::Tensor>& block_mask) {
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

    return ttnn::prim::sol_attn(
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
        false,         // use_mla
        std::nullopt,  // head_dim_v
        memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        std::move(program_config),
        kernel_config_val,
        cu_window_seqlens,
        windowed_q_token_offset,
        windowed_q_token_offset_tensor,
        std::nullopt,  // paged_cache_geometry
        chunk_keep_every,
        block_mask);
}

}  // namespace ttnn::transformer
