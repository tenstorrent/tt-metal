// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Per-NoC trid ring for split block-tile gathers. Reader and writer each issue one half of a block's K/V tile
// reads, capped at K_TRID_RING outstanding transactions per NoC.
// Include after api/dataflow/dataflow_api.h and experimental_device_api.hpp (the reader/writer already do).
#pragma once

#include <stdint.h>

namespace sparse_sdpa_msa {

// Per-NoC trid-ring depth. 0 = off (plain burst). Keep <= 16.
constexpr uint32_t K_TRID_RING = 8;
// Avoid compile-time division by zero when K_TRID_RING is 0.
constexpr uint32_t TRID_MOD = (K_TRID_RING == 0) ? 1u : K_TRID_RING;

// One ring per block gather. K and V reads share the ring so their traffic overlaps.
struct TridRing {
    Noc& noc;
    uint32_t issued = 0;

    // One tile into L1 at `dst_l1` (an absolute address: the block cache's slots do not follow the CB's FIFO
    // pointers).
    template <typename Accessor>
    FORCE_INLINE void read(const Accessor& t, uint32_t tile_bytes, uint32_t page_id, uint32_t dst_l1) {
        if constexpr (K_TRID_RING == 0) {
            noc.async_read(t, CoreLocalMem<uint32_t>(dst_l1), tile_bytes, {.page_id = page_id}, {});
        } else {
            const uint32_t trid = (issued % TRID_MOD) + 1;
            if (issued >= K_TRID_RING) {
                experimental::async_read_barrier_with_trid(noc, trid);  // free this slot before reuse
            }
            experimental::set_read_trid(noc, trid);
            noc.async_read(t, CoreLocalMem<uint32_t>(dst_l1), tile_bytes, {.page_id = page_id}, {});
            ++issued;
        }
    }

    FORCE_INLINE void drain() {
        if constexpr (K_TRID_RING == 0) {
            noc.async_read_barrier();
        } else {
            const uint32_t to_drain = (issued < K_TRID_RING) ? issued : K_TRID_RING;
            for (uint32_t d = 0; d < to_drain; ++d) {
                experimental::async_read_barrier_with_trid(noc, ((issued - to_drain + d) % TRID_MOD) + 1);
            }
            experimental::set_read_trid(noc, 0);  // restore untagged
            issued = 0;
        }
    }
};

}  // namespace sparse_sdpa_msa
