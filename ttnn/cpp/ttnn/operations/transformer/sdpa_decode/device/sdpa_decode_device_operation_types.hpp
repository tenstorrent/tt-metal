// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
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
    // Paired paged-cache geometry overrides (shared with chunked prefill SDPA). See
    // ttnn::operations::transformer::PagedCacheGeometryOverride.
    ttnn::operations::transformer::PagedCacheGeometryOverride paged_cache_geometry{};
    // Optional circular-buffer capacity (in tokens) for the K/V cache view. When set,
    // the kernel computes ``virtual_pos %= cache_position_modulo`` before resolving the
    // page_table entry for every tile read, so a bounded sliding-window cache of
    // capacity N can be addressed by absolute positions ≥ N. Required for layers using
    // vLLM's SlidingWindowSpec, which sizes the per-layer page_table to
    // sliding_window/block_size entries and zero-pads the rest. Must be a multiple of
    // the effective block_size and ≥ sliding_window_size when both are set. Paged-mode
    // only (validated in validate_on_program_cache_miss).
    std::optional<uint32_t> cache_position_modulo = std::nullopt;

    uint8_t share_cache_hash_tag() const {
        return share_cache.has_value() ? (share_cache.value() ? uint8_t{2} : uint8_t{1}) : uint8_t{0};
    }

    static constexpr auto attribute_names = std::forward_as_tuple(
        "scale",
        "output_mem_config",
        "program_config",
        "compute_kernel_config",
        "k_chunk_size",
        "paged_attention",
        "is_causal",
        "share_cache_hash_tag",
        "cur_pos",
        "use_mla",
        "head_dim_v",
        "sliding_window_size",
        "block_size_override",
        "num_kv_heads_override",
        "cache_position_modulo");
    auto attribute_values() const {
        return std::make_tuple(
            std::cref(scale),
            std::cref(output_mem_config),
            std::cref(program_config),
            std::cref(compute_kernel_config),
            k_chunk_size,
            paged_attention,
            is_causal,
            share_cache_hash_tag(),
            std::cref(cur_pos),
            std::cref(use_mla),
            std::cref(head_dim_v),
            std::cref(sliding_window_size),
            std::cref(block_size_override),
            std::cref(num_kv_heads_override),
            std::cref(cache_position_modulo));
    }
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

    tt::tt_metal::Shape q_padded_shape;
    tt::tt_metal::Shape k_padded_shape;
    std::optional<tt::tt_metal::Shape> v_padded_shape;
    std::optional<tt::tt_metal::Shape> cur_pos_tensor_logical_shape;
};

}  // namespace ttnn::prim
