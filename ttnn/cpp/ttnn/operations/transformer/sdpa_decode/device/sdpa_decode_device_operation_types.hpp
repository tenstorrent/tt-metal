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
    // B rows, and cur_pos_tensor holds B*Tg positions, ascending within each group of Tg —
    // candidate b*Tg+j attends to [0, cur_pos[b*Tg+j]] inclusive, exactly like batch row
    // b*Tg+j with that cur_pos in the legacy B*Tg-row form. B groups let the op fill a grid
    // that one group's row-tile count cannot (L1 caps cores/head by Tg).
    // Requires num_kv_heads==1, causal, paged, no sliding window, no MLA, unsharded bf16
    // TILE Q with Tg*32 padded rows per batch.
    uint32_t spec_multi_pos_tiles = 0;
    // Packed speculative mode, on top of spec_multi_pos_tiles = Tg. 0 = off (the Tg-row-tile
    // layout above, byte for byte). When set to NH, the number of valid q rows (heads) at the
    // top of each candidate row-tile, the op packs a group's Tg candidates x NH heads into ONE
    // 32-row q tile: candidate j's heads land on rows [j*S, j*S + NH), S = NH rounded up to an
    // even row count (keeps every row block 64 B aligned for the BH DRAM reads). The compute
    // then runs one row-tile per KV chunk instead of Tg, and the causal mask is built per row
    // (candidate j's rows stop at cur_pos[b*Tg + j]). The KV chunking, the core split and the
    // reduction order are those of the unpacked call, so the valid output rows are
    // bit-identical to it. Q and the output keep the unpacked [1, B, Tg*32, DH] layout; only
    // rows [0, NH) of each output row-tile are written (the padding rows NH..31 are left as
    // allocated). Requires Tg * S <= 32 and an interleaved output.
    uint32_t spec_pack_heads = 0;
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
