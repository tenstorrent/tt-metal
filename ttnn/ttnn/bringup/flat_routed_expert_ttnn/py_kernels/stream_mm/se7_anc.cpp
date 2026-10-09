// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader-less spatial expert: core A's NCRISC (NOC1). Reads A's own gate/up weight blocks (a contiguous region of one
// DRAM bank, one read per block of SLOT_TILES tiles) straight into its in1 ring: block n goes to slot n % SLOTS
// once A's compute has freed it and partner B has copied the block that was there (BCOPY), BATCH blocks per read
// barrier; then tells B it is there (AVAIL). On a chain head it also reads the group's x blocks from DRAM into this
// core's x ring (as se4_xhead.cpp), interleaved with the weight reads.
//
// CT: 0 IN1_CB, 1 SLOT_TILES, 2 W_TILE_BYTES, 3 TOTAL_W_BLOCKS, 4 SLOTS, 5 AVAIL_SEM (on B), 6 BCOPY_SEM, 7
// X_BLK_TILES,
//     8 X_TILE_BYTES, 9 X_BLOCKS, 10 X_SLOTS, 11 XARR_SEM, 12 HFREE_SEM, 13 G, 14 X_PIECES, 15 IS_BFP8, 16 BATCH
// RT: 0 partner (B) xy, 1 weight bank base address, 2 bank id, 3 is chain head, 4 x DRAM address,
//     5 x ring address, 6 group, 7 byte offset of this core's region in the bank
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
    constexpr bool is_bfp8 = get_compile_time_arg_val(15) != 0;
    constexpr uint32_t batch = get_compile_time_arg_val(16);

    const uint32_t bxy = get_arg_val<uint32_t>(0);
    const uint64_t wsrc =
        get_noc_addr_from_bank_id<true>(get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(1) + get_arg_val<uint32_t>(7));
    const bool head = get_arg_val<uint32_t>(3) != 0;
    const InterleavedAddrGenFast<true> x = {
        .bank_base_address = get_arg_val<uint32_t>(4), .page_size = x_tile, .data_format = DataFormat::Bfp8_b};
    const uint32_t x_ring = get_arg_val<uint32_t>(5);
    const uint32_t group = get_arg_val<uint32_t>(6);
    auto sem = [](uint32_t id) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id)); };
    volatile tt_l1_ptr uint32_t* bcopy = sem(bcopy_sem_id);
    volatile tt_l1_ptr uint32_t* xarr = sem(xarr_sem_id);
    volatile tt_l1_ptr uint32_t* hfree = sem(hfree_sem_id);
    const uint64_t b_avail = get_noc_addr(bxy >> 16, bxy & 0xFFFF, get_semaphore(avail_sem_id));
    const uint32_t ring = get_write_ptr(in1_cb);  // the ring's base (nothing pushed yet)

    uint32_t w = 0, xn = 0;
    const uint32_t x_total = head ? x_blocks : 0;
    while (w < total_w || xn < x_total) {
        invalidate_l1_cache();
        uint32_t k = 0;
        while (k < batch && w + k < total_w && cb_pages_reservable_at_back(in1_cb, (k + 1) * slot) &&
               (w + k < slots || *bcopy >= w + k + 1 - slots)) {
            const uint32_t l1 = ring + ((w + k) % slots) * slot * w_tile;
            noc_async_read(wsrc + (w + k) * slot * w_tile, l1, slot * w_tile);
            ++k;
        }
        if (k) {
            noc_async_read_barrier();
            for (uint32_t i = 0; i < k; ++i) {
                cb_push_back(in1_cb, slot);
            }
            w += k;
            noc_semaphore_inc(b_avail, k);
        }
        uint32_t kx = 0;  // up to 6 x blocks per read barrier (DRAM latency)
        while (kx < 6 && xn + kx < x_total && (xn + kx < x_slots || *hfree >= xn + kx + 1 - x_slots)) {
            const uint32_t l1 = x_ring + ((xn + kx) % x_slots) * x_blk * x_tile;
            const uint32_t pg = ((xn + kx) * groups + group) * x_blk;
            for (uint32_t t = 0; t < x_blk; ++t) {
                noc_async_read_tile(pg + t, x, l1 + t * x_tile);
            }
            ++kx;
        }
        if (kx) {
            noc_async_read_barrier();
            xn += kx;
            *xarr = xn * xp;  // the chain counts pieces
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
