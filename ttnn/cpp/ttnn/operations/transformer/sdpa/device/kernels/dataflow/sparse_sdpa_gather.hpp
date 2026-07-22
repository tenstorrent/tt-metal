// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared helper for the dual-NoC K gather. The reader and writer each gather one half of every K chunk on
// their own (factory-assigned) NoC into the same shared cb_k_rm L1; both halves use this trid ring. Include
// after api/dataflow/dataflow_api.h and experimental_device_api.hpp (the reader/writer already do).
#pragma once

#include <stdint.h>

#include "block_cyclic_remap.hpp"  // tt::block_cyclic::logical_to_physical_page (invP remap, rows)

namespace sparse_sdpa {

// Per-NoC trid-ring depth for each gather half (swept: 4 is the sweet spot — with the split halving each
// NoC's load, a shallow ring beats both the plain burst and the old deep single-NoC ring).
constexpr uint32_t K_TRID_RING = 4;

template <
    bool BlockCyclic,
    uint32_t ChunkLocal,
    uint32_t Sp,
    uint32_t ShardStrideGap,
    uint32_t SlabStrideGap,
    typename Accessor>
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
        const uint32_t page =
            tt::block_cyclic::logical_to_physical_page<BlockCyclic, ChunkLocal, Sp, ShardStrideGap, SlabStrideGap>(
                idx_ptr[base + p]);
        noc.async_read(kv, k_cb, k_row_bytes, {.page_id = page_offset + page}, {.offset_bytes = p * k_row_bytes});
    }
    const uint32_t to_drain = (cnt < D) ? cnt : D;
    for (uint32_t d = 0; d < to_drain; ++d) {
        experimental::async_read_barrier_with_trid(noc, ((cnt - to_drain + d) % D) + 1);
    }
    experimental::set_read_trid(noc, 0);  // restore untagged
}

}  // namespace sparse_sdpa
