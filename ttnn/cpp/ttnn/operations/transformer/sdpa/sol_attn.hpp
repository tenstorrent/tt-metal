// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include <optional>

namespace ttnn::transformer {

// Sol-Attn, forked from the non-ring scaled_dot_product_attention.
//
// Currently an exact clone of its parent. It sits where vsa_sdpa does -- after the K/V all-gather --
// because that is the slot a block-sparse fine stage occupies; the ring op solves a delivery problem
// the gather has already solved. The parent was chosen over vsa_sdpa because vsa_sdpa spends 103.7 ms
// on the attention production's dense op (RingJointSDPADeviceOperation, measured 51.47 ms) does in
// half that, so its streaming design carries a ~2x structural penalty a consumer would inherit.
//
// What gets added on top, in order: chunk skipping, a block mask through the provided-mask path this
// op already drives, and the pooled-summary residual sharing one row max with the exact branch.
ttnn::Tensor sol_attn(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const std::optional<ttnn::Tensor>& attn_mask = std::nullopt,
    bool is_causal = true,  // mirrors the parent's binding default
    std::optional<float> scale = std::nullopt,
    std::optional<uint32_t> sliding_window_size = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<ttnn::Tensor>& attention_sink = std::nullopt,
    const std::optional<ttnn::Tensor>& cu_window_seqlens = std::nullopt,
    uint32_t windowed_q_token_offset = 0,
    const std::optional<ttnn::Tensor>& windowed_q_token_offset_tensor = std::nullopt,
    // experiment knob: keep every Nth KV chunk, skip the rest. 0 or 1 = dense.
    uint32_t chunk_keep_every = 0,
    // Per-(q_chunk, k_chunk) keep/skip byte map: [B|1, NQH|1, q_chunks, k_chunks], UINT8, ROW_MAJOR,
    // DRAM. 0 skips the k_chunk's K/V DRAM read entirely. Non-causal only; mutually exclusive with
    // chunk_keep_every.
    const std::optional<ttnn::Tensor>& block_mask = std::nullopt);

}  // namespace ttnn::transformer
