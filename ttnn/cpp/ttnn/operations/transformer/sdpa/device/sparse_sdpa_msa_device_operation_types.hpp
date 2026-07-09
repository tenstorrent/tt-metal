// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "block_cyclic_layout.hpp"  // ttnn::prim::BlockCyclicLayout (shared with sparse_sdpa)
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
    // SP mesh axis used to derive the per-device chunk_start (chunk_start_idx + rank*S); host-side only.
    std::optional<uint32_t> cluster_axis = std::nullopt;
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
