// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::prim {

// Block-cyclic ("slab") KV layout, shared by sparse_sdpa, sparse_sdpa_msa, and indexer_score. When set, the gathered
// `indices` are NATURAL positions but the kv cache is stored block-cyclic across `sp` SP shards with per-shard
// chunk `chunk_local` (the chunked-prefill cache, written by update_padded_kv_cache). The gather kernels remap
// each natural index -> physical page/block on the fly (invP, the inverse of the writer's blockcyclic_positions),
// so the host does NOT reorder the kv buffer back to natural order before the op.
struct BlockCyclicLayout {
    uint32_t sp;           // SP shard count the cache was written across (resolved from the mesh)
    uint32_t chunk_local;  // per-shard chunk length (chunk_size_global / sp == per-chip seq_len_local)
};

}  // namespace ttnn::prim
