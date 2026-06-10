// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "../zero_padded_kv_cache_common.hpp"

// Writes this chip's share of the pad window back to the cache: the masked partial tile (from the
// compute out CB) goes back to the boundary seq-tile, and every fully-pad seq-tile is overwritten
// with zeros from a pre-zeroed L1 scratch buffer. The owned tiles are contiguous-local, so the full
// tiles form a contiguous page range.
void kernel_main() {
    constexpr uint32_t out_cb = get_compile_time_arg_val(0);
    constexpr uint32_t zero_cb = get_compile_time_arg_val(1);
    constexpr uint32_t cache_tile_bytes = get_compile_time_arg_val(2);
    constexpr auto cache_args = TensorAccessorArgs<3>();

    const uint32_t cache_addr = get_arg_val<uint32_t>(0);

    const ZeroPadChipWork w = zero_pad_compute_chip_work();
    if (w.count == 0) {
        return;  // nothing on this chip
    }

    const auto s = TensorAccessor(cache_args, cache_addr, cache_tile_bytes);
    const uint32_t base_page = w.batch_page_base + w.base_local_tile * w.Wt;

    uint32_t first_full_tile = 0;
    if (w.first_partial) {
        // Write the masked partial seq-tile (Wt width-tiles) back to the boundary tile.
        cb_wait_front(out_cb, w.Wt);
        uint32_t l1 = get_read_ptr(out_cb);
        for (uint32_t i = 0; i < w.Wt; ++i) {
            noc_async_write(l1, s.get_noc_addr(base_page + i), cache_tile_bytes);
            l1 += cache_tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(out_cb, w.Wt);
        first_full_tile = 1;
    }

    if (first_full_tile >= w.count) {
        return;  // only a partial tile on this chip, no full pad tiles
    }

    // Pre-zero a one-tile L1 scratch, then write it to every full pad page.
    cb_reserve_back(zero_cb, 1);
    const uint32_t zptr = get_write_ptr(zero_cb);
    volatile tt_l1_ptr uint32_t* zw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zptr);
    for (uint32_t b = 0; b < cache_tile_bytes / 4; ++b) {
        zw[b] = 0;
    }
    cb_push_back(zero_cb, 1);

    const uint32_t full_start_page = base_page + first_full_tile * w.Wt;
    const uint32_t full_end_page = base_page + w.count * w.Wt;
    for (uint32_t page = full_start_page; page < full_end_page; ++page) {
        noc_async_write(zptr, s.get_noc_addr(page), cache_tile_bytes);
    }
    noc_async_write_barrier();
}
