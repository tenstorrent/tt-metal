// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <optional>

namespace ttnn::prim {

struct SolAttnParams {
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
    // Chunked/paged geometry overrides (shared with paged decode). See
    // ttnn::operations::transformer::PagedCacheGeometryOverride.
    ttnn::operations::transformer::PagedCacheGeometryOverride paged_cache_geometry;

    // Experiment knob, not a serving feature: keep every Nth KV chunk and skip the rest, so the skip
    // machinery can be proven and its payoff measured before a real per-dispatch mask exists.
    uint32_t chunk_keep_every = 0;
};

// Per-(q_chunk, k_chunk) keep/skip decision, one byte each, ROW_MAJOR uint8, DRAM. Value 0 skips the
// k_chunk (no K/V DRAM read, no forwarding, no mask read); anything else keeps it. Row 0 (q_chunk 0)
// and column 0 (k_chunk 0) of every row are read but the reader always treats k_chunk 0 as kept
// regardless of the byte's value, mirroring chunk_keep_every's "chunk 0 always kept" floor so a row
// can never select nothing. Broadcasts over batch/heads the same way attn_mask does: dim 0 size 1
// broadcasts over batch, dim 1 size 1 broadcasts over heads.
struct SolAttnInputs {
    Tensor q;
    Tensor k;
    std::optional<Tensor> v;
    std::optional<Tensor> attn_mask;
    std::optional<Tensor> page_table;
    // Mirrors SolAttnParams::chunk_start_idx_tensor so ProgramDescriptor buffer bindings can patch cache hits.
    std::optional<Tensor> chunk_start_idx_tensor;
    std::optional<Tensor> attention_sink;
    // Cumulative window sequence lengths [num_windows + 1], int32/uint32, ROW_MAJOR. Present only in
    // windowed mode; the writer builds the block-diagonal mask from it.
    std::optional<Tensor> cu_window_seqlens;
    // Windowed mode: 1-element int32/uint32 ROW_MAJOR tensor holding the Q shard's global row origin,
    // read by the writer at runtime. Present only when the caller wants a per-device offset under one
    // shared program; otherwise the scalar windowed_q_token_offset is used.
    std::optional<Tensor> windowed_q_token_offset_tensor;
    // Per-(q_chunk, k_chunk) keep/skip byte map. See the comment above SolAttnInputs.
    std::optional<Tensor> block_mask;
};

}  // namespace ttnn::prim
