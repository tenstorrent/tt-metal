// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <array>
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
    // Global row index of Q row 0, when Q is a sequence-parallel shard of a longer sequence. Q and the
    // output are addressed locally; cu_window_seqlens and K/V stay global, so the mask generator offsets
    // Q by this to find the right windows. 0 means Q spans the whole sequence (the unsharded case).
    uint32_t windowed_q_token_offset = 0;
    // 3D-neighborhood (NATTEN) windowed mode: each query attends a (kt,kh,kw) box, inward-shifted at
    // borders, over a (T,H,W) grid flattened T-outer (t = idx/(H*W), h = (idx%(H*W))/W, w = idx%W).
    // When set, the writer synthesizes the per-element 3D mask on-device (no cu_window_seqlens needed);
    // mutually exclusive with the 1D windowed / sliding-window modes. Layout: {T, H, W, kt, kh, kw}.
    std::optional<std::array<uint32_t, 6>> neighborhood_3d;
    // Spatial sequence-parallel over W for neighborhood_3d: the (T,H,W) above is this chip's LOCAL
    // padded shard; this carries {W_full, w_origin} so the mask computes each column's GLOBAL w =
    // w_origin + local_w and clamps the window in [0, W_full). w_origin is a signed int32 stored in a
    // uint32 (a left-edge shard's fake halo maps to negative global w). Absent => not W-sharded.
    std::optional<std::array<uint32_t, 2>> neighborhood_w_shard;
    // Fused-gather variant of neighborhood_3d: instead of streaming the box's active K/V tiles from a
    // TILE-layout K/V and masking per tile, the reader densely gathers each query chunk's window rows
    // from a ROW_MAJOR K/V table into a contiguous cb_k/cb_v (row-granular), so the compute runs dense
    // flash over only real window tokens. Only meaningful when neighborhood_3d is set. Off => the
    // existing streamed-active-tile path. (Build-out in progress; off by default.)
    bool neighborhood_gather = false;
    // Chunked/paged geometry overrides (shared with paged decode). See
    // ttnn::operations::transformer::PagedCacheGeometryOverride.
    ttnn::operations::transformer::PagedCacheGeometryOverride paged_cache_geometry;
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
    // Windowed mode: 1-element int32/uint32 ROW_MAJOR tensor holding the Q shard's global row origin,
    // read by the writer at runtime. Present only when the caller wants a per-device offset under one
    // shared program; otherwise the scalar windowed_q_token_offset is used.
    std::optional<Tensor> windowed_q_token_offset_tensor;
};

}  // namespace ttnn::prim
