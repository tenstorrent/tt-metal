// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <optional>

namespace ttnn::prim {

struct SDPAParams {
    std::optional<float> scale;
    tt::tt_metal::MemoryConfig output_mem_config;
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config;
    bool is_causal = false;
    std::optional<int64_t> chunk_start_idx;        // Chunked legacy: scalar offset, part of program cache key
    std::optional<Tensor> chunk_start_idx_tensor;  // Chunked flexible: device tensor [1] int32, read at runtime
    DeviceComputeKernelConfig compute_kernel_config;
    bool use_mla = false;
    std::optional<uint32_t> head_dim_v;
    std::optional<uint32_t> sliding_window_size;
    // Windowed (block-diagonal) attention: when true, the mask is synthesized on-device from the
    // cu_window_seqlens tensor instead of being read from attn_mask. Implies non-causal.
    bool is_windowed = false;
    // Chunked/paged geometry overrides (mirrors paged_scaled_dot_product_attention_decode).
    // When the paged K/V cache was allocated for a *different* layer's (num_kv_heads,
    // block_size, head_dim) shape — e.g. vLLM's hybrid kv-cache-groups HMA-shares one physical
    // buffer between a full-attention layer (head_dim=512) and a sliding layer (head_dim=256) —
    // the cache's declared shape no longer matches this call's view. Q's last dim drives
    // head_dim; these overrides supply the view's block_size / num_kv_heads so the reader
    // addresses the buffer correctly (the per-block element count must be invariant). Unset =
    // use the cache tensor's own declared block_size / num_kv_heads (the common case).
    std::optional<uint32_t> block_size_override;
    std::optional<uint32_t> num_kv_heads_override;
};

struct SDPAInputs {
    Tensor q;
    Tensor k;
    std::optional<Tensor> v;
    std::optional<Tensor> attn_mask;
    std::optional<Tensor> page_table;
    // Mirrors SDPAParams::chunk_start_idx_tensor so ProgramDescriptor buffer bindings can patch cache hits.
    std::optional<Tensor> chunk_start_idx_tensor;
    std::optional<Tensor> attention_sink;
    // Cumulative window sequence lengths [num_windows + 1], int32/uint32, ROW_MAJOR. Present only in
    // windowed mode; the writer builds the block-diagonal mask from it.
    std::optional<Tensor> cu_window_seqlens;
};

}  // namespace ttnn::prim
