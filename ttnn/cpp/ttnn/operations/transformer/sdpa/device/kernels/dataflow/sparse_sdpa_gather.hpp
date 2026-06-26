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
// invP (the inverse of blockcyclic_positions). Algebraically reduced to one quotient + one runtime multiply:
//   q = n / BC_CHUNK_LOCAL;  slab = q / BC_SP;  c = q % BC_SP
//   page = n + c*bc_delta - slab*BC_K
// BC_CHUNK_LOCAL (= chunk_size/sp) and BC_SP (= sp) are compile-time factory defines, so q folds to mul+shift
// and slab/c to shift+mask (power-of-2 sp); BC_K = (chunk_size/sp)*(sp-1) is also compile-time. Only bc_delta
// (= (T-chunk_size)/sp, runtime because T is) survives, as a single multiplier — the factory precomputes it.
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
    uint32_t page_offset,
    [[maybe_unused]] uint32_t bc_delta = 0) {
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
        // natural -> block-cyclic physical (invP), reduced to page = n + c*delta - slab*K. Only the compile-time
        // divide (q, folds to mul+shift) and one runtime multiply (c*bc_delta) remain.
        const uint32_t q = page / BC_CHUNK_LOCAL;
        const uint32_t slab = q / BC_SP;
        const uint32_t c = q - slab * BC_SP;
        page = page + c * bc_delta - slab * BC_K;
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
