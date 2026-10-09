// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Core B's NCRISC (NOC1) in the paired M-split expert: B shares every weight block with its adjacent partner A, which
// the forwarder only writes once (into A). For each block n, once A has it (AVAIL, incremented by A's se5_recv.cpp) and
// B's in1 ring has room, reads it from A's landing slot n % SLOTS into B's ring, pushes it to B's compute and reports
// the copy (A's BCOPY word) so A can let the slot be refilled. On a chain head it also reads the group's x blocks from
// DRAM into this core's x ring (as se4_xhead.cpp), interleaved with the weight copies.
//
// CT: 0 IN1_CB, 1 SLOT_TILES, 2 W_TILE_BYTES, 3 TOTAL_W_BLOCKS, 4 SLOTS, 5 AVAIL_SEM, 6 BCOPY_SEM, 7 X_BLK_TILES,
//     8 X_TILE_BYTES, 9 X_BLOCKS, 10 X_SLOTS, 11 XARR_SEM, 12 HFREE_SEM, 13 G, 14 X_PIECES
// RT: 0 partner (A) xy, 1 landing ring address, 2 is chain head, 3 x DRAM address, 4 x ring address, 5 group
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t in1_cb = get_compile_time_arg_val(0);
    constexpr uint32_t slot = get_compile_time_arg_val(1);
    constexpr uint32_t w_tile = get_compile_time_arg_val(2);
    constexpr uint32_t total_w = get_compile_time_arg_val(3);
    constexpr uint32_t slots = get_compile_time_arg_val(4);
    constexpr uint32_t avail_sem_id = get_compile_time_arg_val(5);
    constexpr uint32_t bcopy_sem_id = get_compile_time_arg_val(6);
    constexpr uint32_t x_blk = get_compile_time_arg_val(7);
    constexpr uint32_t x_tile = get_compile_time_arg_val(8);
    constexpr uint32_t x_blocks = get_compile_time_arg_val(9);
    constexpr uint32_t x_slots = get_compile_time_arg_val(10);
    constexpr uint32_t xarr_sem_id = get_compile_time_arg_val(11);
    constexpr uint32_t hfree_sem_id = get_compile_time_arg_val(12);
    constexpr uint32_t groups = get_compile_time_arg_val(13);
    constexpr uint32_t xp = get_compile_time_arg_val(14);
    constexpr uint32_t slot_bytes = slot * w_tile;

    const uint32_t axy = get_arg_val<uint32_t>(0);
    const uint32_t land = get_arg_val<uint32_t>(1);
    const bool head = get_arg_val<uint32_t>(2) != 0;
    const InterleavedAddrGenFast<true> x = {
        .bank_base_address = get_arg_val<uint32_t>(3), .page_size = x_tile, .data_format = DataFormat::Bfp8_b};
    const uint32_t x_ring = get_arg_val<uint32_t>(4);
    const uint32_t group = get_arg_val<uint32_t>(5);
    auto sem = [](uint32_t id) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id)); };
    volatile tt_l1_ptr uint32_t* avail = sem(avail_sem_id);
    volatile tt_l1_ptr uint32_t* xarr = sem(xarr_sem_id);
    volatile tt_l1_ptr uint32_t* hfree = sem(hfree_sem_id);
    const uint64_t a_land = get_noc_addr(axy >> 16, axy & 0xFFFF, land);
    const uint64_t a_bcopy = get_noc_addr(axy >> 16, axy & 0xFFFF, get_semaphore(bcopy_sem_id));

    uint32_t w = 0, xn = 0;
    const uint32_t x_total = head ? x_blocks : 0;
    while (w < total_w || xn < x_total) {
        invalidate_l1_cache();
        // Up to MAX_BATCH blocks per read barrier (one block at a time is latency-bound at ~11 GB/s). B's ring is the
        // landing tensor too, so block n goes to slot n % SLOTS at the same address as on A.
        constexpr uint32_t max_batch = 4;
        uint32_t k = 0;
        while (k < max_batch && w + k < total_w && *avail > w + k &&
               cb_pages_reservable_at_back(in1_cb, (k + 1) * slot)) {
            const uint32_t off = ((w + k) % slots) * slot_bytes;
            noc_async_read(a_land + off, land + off, slot_bytes);
            ++k;
        }
        if (k) {
            noc_async_read_barrier();
            for (uint32_t i = 0; i < k; ++i) {
                cb_push_back(in1_cb, slot);
            }
            w += k;
            noc_semaphore_inc(a_bcopy, k);  // atomic delta (an inline write would wait for all writes to drain)
        }
        uint32_t kx = 0;  // up to 3 x blocks per read barrier (DRAM latency)
        while (kx < 6 && xn + kx < x_total && (xn + kx < x_slots || *hfree >= xn + kx + 1 - x_slots)) {
            const uint32_t l1 = x_ring + ((xn + kx) % x_slots) * x_blk * x_tile;
            const uint32_t page0 = ((xn + kx) * groups + group) * x_blk;
#ifndef SE_FAKE_X  // diagnostic: skip the DRAM read (wrong results, timing only)
            for (uint32_t t = 0; t < x_blk; ++t) {
                noc_async_read_tile(page0 + t, x, l1 + t * x_tile);
            }
#endif
            ++kx;
        }
        if (kx) {
            noc_async_read_barrier();
            xn += kx;
            *xarr = xn * xp;  // the chain counts pieces
        }
    }
    noc_async_write_barrier();
}
