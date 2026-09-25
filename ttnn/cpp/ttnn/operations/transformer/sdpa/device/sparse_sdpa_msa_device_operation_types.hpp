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
    // SP mesh axis the query sequence is sharded over. Its rank selects the device's slab; with a block-cyclic
    // layout it also makes the causal geometry rotation-exact (see block_cyclic_causal_geometry.hpp), exactly as
    // indexer_score_msa's seq_shard_axes=[sp]. Host-side for chunk_start_idx; a compile-time flag on the
    // metadata path.
    std::optional<uint32_t> cluster_axis = std::nullopt;
    // Layer fold for the trace-safe slot select (see SparseSDPAMsaInputs::cache_batch_idx_tensor): the slot is
    // user_id * index_cache_num_layers + index_cache_layer_idx. Runtime, NOT hashed -- one program serves every
    // user and layer, matching the indexer.
    uint32_t index_cache_num_layers = 1;
    uint32_t index_cache_layer_idx = 0;
    bool has_block_cyclic() const { return block_cyclic.has_value(); }
};

struct SparseSDPAMsaInputs {
    Tensor q;        // [1,H,S,d] bf16|fp8_e4m3 ROW_MAJOR
    Tensor k;        // [B,n_kv,T,d] TILE bf16|bfp8_b
    Tensor v;        // [B,n_kv,T,v_dim] TILE bf16|bfp8_b
    Tensor indices;  // [1,n_kv,S,TOPK] uint32 block ids; 0xFFFFFFFF is the sentinel
    // TRACE-SAFE metadata: 1-element UINT32 row-major interleaved DRAM tensors the reader (and, for the slot, the
    // writer) NoC-read on every dispatch. A host scalar is patched into the launch per dispatch and a trace replay
    // never re-runs that patch, so a captured program would keep the capture-time slot / depth.
    // chunk_start_idx_tensor: rank 0's global start (replaces chunk_start_idx; enables causal masking); each
    // device derives its own start and rotation in-kernel.
    std::optional<Tensor> chunk_start_idx_tensor = std::nullopt;
    // cache_batch_idx_tensor: the USER id; the kernels recompose the K/V slot (replaces cache_batch_idx).
    std::optional<Tensor> cache_batch_idx_tensor = std::nullopt;
    bool has_chunk_start_metadata() const { return chunk_start_idx_tensor.has_value(); }
    bool has_cache_slot_metadata() const { return cache_batch_idx_tensor.has_value(); }
};

// Causal masking is on when EITHER chunk-start form is supplied.
inline bool causal_enabled(const SparseSDPAMsaParams& attrs, const SparseSDPAMsaInputs& t) {
    return attrs.chunk_start_idx.has_value() || t.has_chunk_start_metadata();
}
// A cache slot is selected by EITHER the host scalar or the trace-safe tensor.
inline bool selects_cache_slot(const SparseSDPAMsaParams& attrs, const SparseSDPAMsaInputs& t) {
    return attrs.cache_batch_idx.has_value() || t.has_cache_slot_metadata();
}
// Rotation-exact block-cyclic geometry: the SP axis is named (the indexer's seq_shard_axes=[sp] predicate).
inline bool rotation_exact_geometry(const SparseSDPAMsaParams& attrs) { return attrs.cluster_axis.has_value(); }

}  // namespace ttnn::prim
