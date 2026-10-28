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
    // Speculative multi-position mode. 0 = off (legacy behaviour, byte-for-byte).
    // When set to Tg (candidates per batch GROUP), the op folds Tg candidates that share one
    // KV cache onto a single batch row, so each row scans its KV once instead of Tg times:
    // Q is [1, B, Tg*32, DH] (batch b, row-tile j = candidate b*Tg + j), the page_table has
    // B rows, and cur_pos_tensor holds B*Tg positions, ascending within each group of Tg apart
    // from -1 entries — candidate b*Tg+j attends to [0, cur_pos[b*Tg+j]] inclusive, exactly like
    // batch row b*Tg+j with that cur_pos in the legacy B*Tg-row form; a candidate at -1 gets
    // output 0 (and lse -1e30 with return_lse). B groups let the op fill a grid that one group's
    // row-tile count cannot (L1 caps cores/head by Tg).
    // Requires num_kv_heads==1, causal, paged, no sliding window, MLA only with its own V cache,
    // unsharded bf16 TILE Q with Tg*32 padded rows per batch, an interleaved output and cache
    // blocks of whole tiles.
    uint32_t spec_multi_pos_tiles = 0;
    // Spec mode, Tg even: candidates of at most 16 heads (rows 0..15 of their Q tiles) are computed two
    // per tile (candidate 2t + h in rows 16h .. 16h + 15 of compute row-tile t), half the QK / PV work;
    // Q, the output and lse keep a tile per candidate (output rows 16..31 repeat rows 0..15, lse rows
    // 16..31 are -1e30).
    bool spec_multi_pos_packed = false;
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

    // Optional second output, preallocated by sdpa_decode_with_lse: the log-sum-exp of every (user, Q head),
    // fp32 [1, B, nh, 1] TILE (one tile row per head, the value in column 0, padding 0), in the
    // kernel's logit space: lse = scale * m + log(l). Absent (the default) leaves the op's programs unchanged.
    std::optional<Tensor> lse_output;
};

}  // namespace ttnn::prim
