// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared helper for the dual-NoC K gather. The reader and writer each gather one half of every K chunk on
// their own (factory-assigned) NoC into the same shared cb_k_rm L1; both halves use this trid ring. Include
// after api/dataflow/dataflow_api.h and experimental_device_api.hpp (the reader/writer already do).
#pragma once

#include <stdint.h>

namespace sparse_sdpa {

// Per-NoC trid-ring depth for each gather half (swept: 4 is the sweet spot — with the split halving each
// NoC's load, a shallow ring beats both the plain burst and the old deep single-NoC ring).
constexpr uint32_t K_TRID_RING = 4;

// Gather the K rows for chunk index positions [lo, hi) on `noc`, capping outstanding reads with a depth-D
// trid ring: idx_ptr[base + p] -> k_cb at byte offset p * k_row_bytes, for p in [lo, hi). Used by the
// reader for its half [half, valid) and by the writer for its half [0, half). `page_offset` selects the
// batch slot of an indexed KV cache (cache_batch_idx * T); 0 for a single [1,1,T,K_DIM] cache.
//
// When BC_ENABLE is defined, the kv cache is stored block-cyclic across SP shards (the DeepSeek
// chunked-prefill KVPE cache), so the natural-position index n is remapped to its physical page on the fly via
// invP (the inverse of blockcyclic_positions). Natural order is slab-major (chunk after chunk, shard-within);
// the physical buffer is shard-major (each shard's whole region, slabs-within). Decompose n into its block,
// then re-encode by adjusting the per-shard and per-slab strides:
//   block_idx = n / BC_CHUNK_LOCAL;  slab = block_idx / BC_SP;  shard = block_idx % BC_SP
//   page = n + shard*BC_SHARD_STRIDE_GAP - slab*BC_SLAB_STRIDE_GAP
// All factory compile-time defines (the cache length T is folded into the program hash for this path, so the
// whole remap is compile-time-constant arithmetic — no runtime arg):
//   BC_SP               = sp (number of SP shards)
//   BC_CHUNK_LOCAL      = chunk_local (rows one chunk writes into one shard = chunk_size_global / sp)
//   BC_SHARD_STRIDE_GAP = shard_len - chunk_local : the physical buffer reserves shard_len (= T/sp) rows per
//                         shard, so consecutive shards sit that much further apart than in natural order
//                         (chunk_local) -> push shard s forward by shard*gap.
//   BC_SLAB_STRIDE_GAP  = chunk_global - chunk_local = chunk_local*(sp-1) : consecutive chunks (slabs) are
//                         chunk_global apart in natural order but only chunk_local apart physically -> pull
//                         slab s back by slab*gap.
// Sentinels (0xFFFFFFFF) never reach here — the reader only gathers the valid (< nv) prefix.
template <typename Accessor>
FORCE_INLINE void trid_ring_gather(
    Noc& noc,
    const Accessor& kv,
    experimental::CB& k_cb,
    volatile tt_l1_ptr uint32_t* idx_ptr,
    uint32_t base,
    uint32_t lo,
    uint32_t hi,
    uint32_t k_row_bytes,
    uint32_t page_offset) {
    constexpr uint32_t D = K_TRID_RING;
    const uint32_t cnt = hi - lo;
    for (uint32_t i = 0; i < cnt; ++i) {
        const uint32_t p = lo + i;
        const uint32_t trid = (i % D) + 1;
        if (i >= D) {
            experimental::async_read_barrier_with_trid(noc, trid);  // free this trid slot before reuse
        }
        experimental::set_read_trid(noc, trid);
        uint32_t page = idx_ptr[base + p];
#ifdef BC_ENABLE
        // natural index n -> block-cyclic physical row (invP). All terms compile-time (T is hashed), so this is
        // one compile-time divide (mul+shift) + shift/mask. See the header comment for the layout reasoning.
        const uint32_t block_idx = page / BC_CHUNK_LOCAL;  // = slab*BC_SP + shard (natural-order block index)
        const uint32_t slab = block_idx / BC_SP;
        const uint32_t shard = block_idx - slab * BC_SP;
        page = page + shard * BC_SHARD_STRIDE_GAP - slab * BC_SLAB_STRIDE_GAP;
#endif
        noc.async_read(kv, k_cb, k_row_bytes, {.page_id = page_offset + page}, {.offset_bytes = p * k_row_bytes});
    }
    const uint32_t to_drain = (cnt < D) ? cnt : D;
    for (uint32_t d = 0; d < to_drain; ++d) {
        experimental::async_read_barrier_with_trid(noc, ((cnt - to_drain + d) % D) + 1);
    }
    experimental::set_read_trid(noc, 0);  // restore untagged
}

}  // namespace sparse_sdpa
