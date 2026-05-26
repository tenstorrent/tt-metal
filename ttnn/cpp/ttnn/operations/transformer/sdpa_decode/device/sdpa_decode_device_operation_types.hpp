// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"

namespace ttnn::prim {

struct SdpaDecodeParams {
    bool is_causal = false;
    bool paged_attention = false;
    std::vector<uint32_t> cur_pos;
    std::optional<float> scale = std::nullopt;
    std::optional<uint32_t> sliding_window_size = std::nullopt;
    tt::tt_metal::MemoryConfig output_mem_config;
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config = std::nullopt;
    DeviceComputeKernelConfig compute_kernel_config;
    uint32_t k_chunk_size = 0;
    // Share cache is only meaningful for some unpaged configurations; default is false.
    std::optional<bool> share_cache = std::nullopt;
    // When true, enables multi-latent attention (MLA) path where V is derived from K.
    std::optional<bool> use_mla = std::nullopt;
    std::optional<uint32_t> head_dim_v = std::nullopt;
    // Optional per-call block_size for paged attention. Lets this call read a K/V cache
    // that was allocated for a different layer's (block_size, head_dim) shape — Q's last
    // dim then drives head_dim. Required when vLLM's shared kv-cache groups place layers
    // with different specs on one physical buffer. num_kv_heads * block_size * head_dim
    // must be preserved across views (checked in validate_on_program_cache_miss).
    std::optional<uint32_t> block_size_override = std::nullopt;
    // Optional per-call num_kv_heads, companion to block_size_override for HMA
    // cross-group sharing where the cache was allocated with a different num_kv_heads
    // from the layer's actual view (e.g. Gemma4-26B-A4B sliding kv=8 / full kv=2 at
    // small TP). Drives both the kernel's per-block stride and its head-parallel
    // reduction grid. When nullopt, falls back to K.padded_shape[1] (legacy behavior).
    std::optional<uint32_t> num_kv_heads_override = std::nullopt;
};

struct SdpaDecodeInputs {
    // Mandatory tensors
    Tensor q;
    Tensor k;

    // Optional V tensor; when MLA is enabled, V is derived from K and this may be nullopt.
    std::optional<Tensor> v;

    // Optional auxiliary tensors
    std::optional<Tensor> cur_pos_tensor;
    std::optional<Tensor> page_table_tensor;
    std::optional<Tensor> attn_mask;
    std::optional<Tensor> attention_sink;
};

}  // namespace ttnn::prim
