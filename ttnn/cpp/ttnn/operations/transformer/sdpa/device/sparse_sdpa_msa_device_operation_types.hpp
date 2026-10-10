// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "block_cyclic_layout.hpp"  // ttnn::prim::BlockCyclicLayout (shared)
#include <optional>

namespace ttnn::prim {

// Primitive for MSA block-sparse prefill. K/V are separate tiled caches; masking comes only from block ids in
// `indices` plus -1 sentinels. For GQA, each KV group owns H/n_kv query heads.
struct SparseSDPAMsaParams {
    float scale = 1.0f;         // compile-time; included in the program hash
    uint32_t block_size = 128;  // tokens per selected KV block
    DeviceComputeKernelConfig compute_kernel_config;
    // Selects one [B,n_kv,T,*] cache slot. The value is patched as runtime K/V tile offsets and is not hashed.
    std::optional<uint32_t> cache_batch_idx = std::nullopt;
    // Set -> the K/V cache is block-cyclic across SP; the gather kernels apply the invP block remap. Hashed.
    std::optional<BlockCyclicLayout> block_cyclic = std::nullopt;
    // Global position of query row 0.
    // Set -> enforce a token-level causal mask on the diagonal block (the query's own block, whose later tokens are
    // future). Unset -> no token-level causality; the op attends the full selected blocks.
    std::optional<uint32_t> chunk_start_idx = std::nullopt;
    // SP mesh axis used to derive the per-device causal geometry; host-side only. chunk_start_idx + rank*S, or,
    // with a block-cyclic cache, the KV writer's rotated position (see compute_causal_geometry).
    std::optional<uint32_t> cluster_axis = std::nullopt;
    // Per-core L1 cache for the gathered K/V blocks: a block re-selected by a later query on the same core is read
    // from L1 instead of DRAM; byte-identical to the streamed kernels. The slot count is resolved per call from the
    // allocator's lowest live L1 address (as many as fit after the op's own CBs, at most KV_CACHE_SLOTS_MAX; fewer
    // than KV_CACHE_SLOTS_MIN runs the streamed kernels) and is part of the program-cache key. It is an estimate,
    // not a reservation: a constraint that mesh-level view cannot see fails the CB/buffer overlap check at launch
    // rather than shrinking the cache, and a trace replays the captured count without that check, so the L1 that
    // was free at capture must be free at replay.
    bool enable_kv_block_cache = false;
    // The slot count resolved against the L1 free at this call, once, at the prim entry; the program hash and the
    // program factory both read it. 0 selects the streamed kernels. Not user-facing.
    uint32_t kv_cache_slots = 0;
    bool has_indexed_kv_cache() const { return cache_batch_idx.has_value(); }
    bool causal_enabled() const { return chunk_start_idx.has_value(); }
    bool has_block_cyclic() const { return block_cyclic.has_value(); }
};

struct SparseSDPAMsaInputs {
    Tensor q;        // [1,H,S,d] bf16|fp8_e4m3 ROW_MAJOR
    Tensor k;        // [B,n_kv,T,d] TILE bf16|bfp8_b
    Tensor v;        // [B,n_kv,T,v_dim] TILE bf16|bfp8_b
    Tensor indices;  // [1,n_kv,S,TOPK] uint32 block ids; 0xFFFFFFFF is the sentinel
};

}  // namespace ttnn::prim
