// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Inverse permutation for a cache striped block-cyclic ("slab") across SP shards. A DeepSeek/MiniMax
// chunked-prefill cache stores, per shard, `[chunk0_local, chunk1_local, ...]`; AllGather over the SP axis
// then yields a shard-major buffer, while the ops index in natural (logical) token order. This maps a
// logical index to its physical position so the gather reads the block-cyclic buffer in place — no host
// reorder. Unit-agnostic (pure index math): each op works in its own granularity — sparse_sdpa in rows,
// sparse_sdpa_msa in blocks, indexer_score in tiles — and bakes chunk_local / both stride gaps as template args.
#pragma once

#include <stdint.h>

namespace tt::block_cyclic {

// invP of the (slab, shard) transpose (see file header). The divisors are template (compile-time) args, so
// this folds to one divide (mul+shift) + shift/mask.
template <uint32_t ChunkLocal, uint32_t Sp, uint32_t ShardStrideGap, uint32_t SlabStrideGap>
FORCE_INLINE uint32_t logical_to_chunked_physical(uint32_t n) {
    const uint32_t block_idx = n / ChunkLocal;  // = slab*Sp + shard
    const uint32_t slab = block_idx / Sp;
    const uint32_t shard = block_idx - slab * Sp;
    return n + shard * ShardStrideGap - slab * SlabStrideGap;
}

// Natural -> physical dispatch. BlockCyclic == false (a contiguous, natural-order cache) folds to the
// identity, so a natural-order cache emits no remap arithmetic.
template <bool BlockCyclic, uint32_t ChunkLocal, uint32_t Sp, uint32_t ShardStrideGap, uint32_t SlabStrideGap>
FORCE_INLINE uint32_t logical_to_physical_page(uint32_t page) {
    if constexpr (BlockCyclic) {
        return logical_to_chunked_physical<ChunkLocal, Sp, ShardStrideGap, SlabStrideGap>(page);
    } else {
        return page;
    }
}

}  // namespace tt::block_cyclic
