// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::prim {

// Block-cyclic ("slab") KV layout, shared by sparse_sdpa (DSA) and sparse_sdpa_msa. When set, the gathered
// `indices` are NATURAL positions but the kv cache is stored block-cyclic across `sp` SP shards with per-shard
// chunk `chunk_local` (the chunked-prefill cache, written by update_padded_kv_cache). The gather kernels remap
// each natural index -> physical page/block on the fly (invP, the inverse of the writer's blockcyclic_positions),
// so the host does NOT reorder the kv buffer back to natural order before the op.
//
// Both fields are resolved+validated at the ttnn entry: `sp` is read from the mesh axis the cache was striped
// over (a caller cannot pass an sp that disagrees with the device), and `chunk_local` is cross-checked against
// q's per-chip seq length. `sp`/`chunk_local` are folded into the program hash (BC_* are compile-time defines),
// so they must be the resolved values, not a mesh axis index.
struct BlockCyclicLayout {
    uint32_t sp;           // SP shard count the cache was written across (resolved from the mesh)
    uint32_t chunk_local;  // per-shard chunk length (chunk_size_global / sp == per-chip seq_len_local)
};

}  // namespace ttnn::prim
