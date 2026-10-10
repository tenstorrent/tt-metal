// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Chained-x DRAM reader (NCRISC, NOC1) on a chain head: reads its M-group's x blocks from the interleaved DRAM tensor
// (order (v, K-block, group), X_BLK_TILES tiles each) straight into this core's x ring, slot n % X_SLOTS, once the
// core has freed that slot (HFREE, written by se4_recv.cpp), then bumps the core's XARR.
//
// CT: 0 X_BLK_TILES, 1 TILE_BYTES, 2 BLOCKS (this group's), 3 X_SLOTS, 4 XARR_SEM, 5 HFREE_SEM, 6 G, 7 X_PIECES
// RT: 0 x DRAM address, 1 x ring address, 2 group
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t blk = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t blocks = get_compile_time_arg_val(2);
    constexpr uint32_t x_slots = get_compile_time_arg_val(3);
    constexpr uint32_t xarr_sem_id = get_compile_time_arg_val(4);
    constexpr uint32_t hfree_sem_id = get_compile_time_arg_val(5);
    constexpr uint32_t groups = get_compile_time_arg_val(6);
    constexpr uint32_t xp = get_compile_time_arg_val(7);
    const InterleavedAddrGenFast<true> x = {
        .bank_base_address = get_arg_val<uint32_t>(0), .page_size = tile_bytes, .data_format = DataFormat::Bfp8_b};
    const uint32_t ring = get_arg_val<uint32_t>(1);
    const uint32_t group = get_arg_val<uint32_t>(2);
    volatile tt_l1_ptr uint32_t* xarr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(xarr_sem_id));
    volatile tt_l1_ptr uint32_t* hfree = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(hfree_sem_id));
    // Up to DEPTH blocks per read barrier: one block at a time is DRAM-latency bound (~8-10 GB/s).
    constexpr uint32_t depth = 6;
    uint32_t n = 0;
    while (n < blocks) {
        invalidate_l1_cache();
        uint32_t k = 0;
        while (k < depth && n + k < blocks && (n + k < x_slots || *hfree >= n + k + 1 - x_slots)) {
            const uint32_t l1 = ring + ((n + k) % x_slots) * blk * tile_bytes;
            const uint32_t page0 = ((n + k) * groups + group) * blk;
#ifndef SE_FAKE_X  // diagnostic: skip the DRAM read (wrong results, timing only)
            for (uint32_t t = 0; t < blk; ++t) {
                noc_async_read_tile(page0 + t, x, l1 + t * tile_bytes);
            }
#endif
            ++k;
        }
        if (k) {
            noc_async_read_barrier();
            n += k;
            *xarr = n * xp;  // the chain counts pieces
        }
    }
}
