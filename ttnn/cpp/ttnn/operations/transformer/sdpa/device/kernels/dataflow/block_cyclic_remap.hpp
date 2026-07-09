// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Inverse permutation for a cache striped block-cyclic ("slab") across SP shards. A DeepSeek/MiniMax
// chunked-prefill cache stores, per shard, `[chunk0_local, chunk1_local, ...]`; AllGather over the SP axis
// then yields a shard-major buffer, while the ops index in natural (logical) token order. This maps a
// logical index to its physical position so the gather reads the block-cyclic buffer in place — no host
// reorder. Shared by sparse_sdpa (rows), sparse_sdpa_msa (blocks), and indexer_score (tiles): the unit is
// the caller's; each passes chunk_local and the two gaps in its own unit.
#pragma once

#include <stdint.h>

namespace tt::block_cyclic {

// Natural order is slab-major (chunk after chunk, shard-within); the physical buffer is shard-major (each
// shard's whole region, slabs-within), so the map is a transpose of the (slab, shard) decomposition:
//   block_idx = n / chunk_local;  slab = block_idx / sp;  shard = block_idx % sp
//   physical  = n + shard*shard_stride_gap - slab*slab_stride_gap
// All args are compile-time constants at every call site, so this folds to one divide (mul+shift) + shift/mask.
inline uint32_t logical_to_chunked_physical(
    uint32_t n, uint32_t chunk_local, uint32_t sp, uint32_t shard_stride_gap, uint32_t slab_stride_gap) {
    const uint32_t block_idx = n / chunk_local;  // = slab*sp + shard
    const uint32_t slab = block_idx / sp;
    const uint32_t shard = block_idx - slab * sp;
    return n + shard * shard_stride_gap - slab * slab_stride_gap;
}

}  // namespace tt::block_cyclic
