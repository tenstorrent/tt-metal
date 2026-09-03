// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <optional>

namespace ttnn::prim {

// Block-cyclic KV layout. When set, the gathered `indices` are NATURAL token positions, but the kv cache is
// stored block-cyclic across `sp` SP shards with per-shard chunk `chunk_local` (the DeepSeek chunked-prefill
// KVPE cache, written by update_padded_kv_cache). The gather kernels remap each natural index -> physical page
// id on the fly (invP, the inverse of the cache writer's blockcyclic_positions), so the host does NOT have to
// reorder the kv buffer back to natural order before the op.
//
// `sp` is resolved from the cache's mesh axis and `chunk_local` is cross-checked against q's local sequence length
// at the ttnn entry. Both are compile-time kernel arguments, so `sp` is the resolved shard count, not a mesh axis.
struct BlockCyclicLayout {
    uint32_t sp;           // SP shard count the cache was written across (resolved from the mesh)
    uint32_t chunk_local;  // per-shard chunk length (chunk_size_global / sp == per-chip seq_len_local)
};

// Sparse MLA prefill (DeepSeek DSA).
struct SparseSDPAParams {
    float scale = 1.0f;  // compile-time (folded into the program hash)
    uint32_t v_dim;      // width of V (= leading v_dim cols of the K_DIM-wide KV cache); the output width
    uint32_t k_chunk_size = 128;
    DeviceComputeKernelConfig compute_kernel_config;
    // Indexed KV cache: when set, kv is a [B,1,T,K_DIM] shared cache and this selects the batch slot to
    // attend to (the gather page ids are offset by cache_batch_idx * T). It is a DYNAMIC runtime arg
    // (excluded from the program hash, re-applied every dispatch), so changing it does NOT recompile.
    std::optional<uint32_t> cache_batch_idx = std::nullopt;
    bool has_indexed_kv_cache() const { return cache_batch_idx.has_value(); }
    // The remap configuration is compile-time; T is part of the program hash for this path.
    std::optional<BlockCyclicLayout> block_cyclic = std::nullopt;
    bool has_block_cyclic() const { return block_cyclic.has_value(); }
};

struct SparseSDPAInputs {
    Tensor q;   // [1, H, S, K_DIM] bf16/fp8_e4m3 ROW_MAJOR  (K_DIM = head dim, e.g. 576)
    Tensor kv;  // [1, 1, T, K_DIM] bf16/fp8_e4m3 ROW_MAJOR  (or [B,1,T,K_DIM] when indexed; may be ND-sharded DRAM)
    Tensor indices;  // [1, 1, S, TOPK] uint32 ROW_MAJOR  (0xFFFFFFFF = masked)
};

}  // namespace ttnn::prim
