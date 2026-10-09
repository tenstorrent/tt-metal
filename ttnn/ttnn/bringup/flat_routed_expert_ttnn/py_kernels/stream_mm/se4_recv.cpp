// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Chained-x M-split big-M receiver data movement (BRISC, NOC0) on a compute core of M-group G. x travels down chains
// of the group's cores by unicast (a chain head's NCRISC reads it from DRAM, se4_xhead.cpp); h still comes from the
// relay's multicast into the group's rectangle. A non-blocking event loop that
//   * runs the in1 ring (one credit per free slot to the forwarder, publishes blocks as they land); the ring holds one
//     expert's weight slice and compute frees blocks after their last use,
//   * publishes x blocks as they arrive (XARR, incremented by the predecessor after the block's write is acknowledged),
//     forwards each block into the successor's x ring slot once the successor has freed it (SFREE, the successor's
//     freed count), then bumps the successor's XARR; a slot counts as freed here once compute has consumed it and it
//     has been forwarded, and that count goes to the predecessor's SFREE (or, on a chain head, to its own HFREE, which
//     its DRAM reader waits on),
//   * sends its h slice of each virtual expert v ([MT x NP] tiles, h K-tiles CG * NP + p) into the relay's buffer of
//     its group with a GATH (group's) increment there, once go >= v + 1 - HBUF,
//   * publishes h_all(v) once the relay's counter (HARR, set by multicast) reaches v / HBUF + 1 (HBUF = 1 here: the
//     relay keeps one buffer per group),
//   * drains each output into an S-deep ring (the last expert's S sub-blocks stay) and reports it to the coordinator,
//     which counts the virtual experts every core has finished ("go").
//
// CT: 0 X_CB, 1 X_BLK_TILES, 2 IN1_CB, 3 SLOT_TILES, 4 TOTAL_W_BLOCKS, 5 NUM_V, 6 W_SLOTS, 7 OUT_CB, 8 OUT_TILES,
//     9 H_LOCAL_CB, 10 H_ALL_CB, 11 MT, 12 H_TILE_BYTES, 13 NCC, 14 DATA_SEM, 15 HARR_SEM, 16 GO_SEM, 17 DONE_SEM,
//     18 KBLK, 19 HARR1_SEM, 20 SFREE_SEM, 21 HBUF, 22 XARR_SEM, 23 HFREE_SEM, 24 X_SLOTS, 25 X_BYTES, 26 S, 27 NP
// RT: 0 forwarder xy, 1 credit sem id on the forwarder, 2 column group CG, 3 relay xy, 4 relay h buffer address (this
//     group's), 5 coordinator xy, 6 group's GATH sem id, 7 x ring address, 8 is coordinator, 9 predecessor xy (0: chain
//     head), 10 successor xy (0: chain tail), 11.. NCC compute-core xy
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t x_cb = get_compile_time_arg_val(0);
    constexpr uint32_t x_blk = get_compile_time_arg_val(1);
    constexpr uint32_t in1_cb = get_compile_time_arg_val(2);
    constexpr uint32_t slot = get_compile_time_arg_val(3);
    constexpr uint32_t total_w = get_compile_time_arg_val(4);
    constexpr uint32_t num_v = get_compile_time_arg_val(5);
    constexpr uint32_t w_slots = get_compile_time_arg_val(6);
    constexpr uint32_t out_cb = get_compile_time_arg_val(7);
    constexpr uint32_t out_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t h_local_cb = get_compile_time_arg_val(9);
    constexpr uint32_t h_all_cb = get_compile_time_arg_val(10);
    constexpr uint32_t mt = get_compile_time_arg_val(11);
    constexpr uint32_t h_tile_bytes = get_compile_time_arg_val(12);
    constexpr uint32_t ncc = get_compile_time_arg_val(13);
    constexpr uint32_t data_sem_id = get_compile_time_arg_val(14);
    constexpr uint32_t harr_sem_id = get_compile_time_arg_val(15);
    constexpr uint32_t go_sem_id = get_compile_time_arg_val(16);
    constexpr uint32_t done_sem_id = get_compile_time_arg_val(17);
    constexpr uint32_t kblk = get_compile_time_arg_val(18);
    constexpr uint32_t harr1_sem_id = get_compile_time_arg_val(19);
    constexpr uint32_t hbuf = get_compile_time_arg_val(21);
    constexpr uint32_t xarr_sem_id = get_compile_time_arg_val(22);
    constexpr uint32_t x_slots = get_compile_time_arg_val(24);
    constexpr uint32_t total_x = get_compile_time_arg_val(25);
    constexpr uint32_t sfree_sem_id = get_compile_time_arg_val(20);
    constexpr uint32_t hfree_sem_id = get_compile_time_arg_val(23);
    constexpr uint32_t x_bytes = get_compile_time_arg_val(25);
    constexpr uint32_t sub = get_compile_time_arg_val(26);
    constexpr uint32_t np = get_compile_time_arg_val(27);
    constexpr uint32_t h_all_tiles = (ncc / 2) * np * mt;  // the group's h: NCC / 2 cores x NP K-tiles, MT rows
    constexpr uint32_t half_bytes = h_all_tiles * h_tile_bytes;
    static_assert(hbuf == 1 || hbuf == 2);

    const uint32_t fxy = get_arg_val<uint32_t>(0);
    const uint32_t credit_sem_id = get_arg_val<uint32_t>(1);
    const uint32_t cg = get_arg_val<uint32_t>(2);
    const uint32_t bxy = get_arg_val<uint32_t>(3);
    const uint32_t bc_h_addr = get_arg_val<uint32_t>(4);
    const uint32_t cxy = get_arg_val<uint32_t>(5);
    const uint32_t gath_sem_id = get_arg_val<uint32_t>(6);
    const uint32_t x_ring = get_arg_val<uint32_t>(7);
    const uint32_t pxy = get_arg_val<uint32_t>(9), sxy = get_arg_val<uint32_t>(10);
    const bool is_coord = get_arg_val<uint32_t>(8) != 0;
    auto sem = [](uint32_t id) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id)); };
    const uint64_t credit_noc = get_noc_addr(fxy >> 16, fxy & 0xFFFF, get_semaphore(credit_sem_id));
    const uint64_t done_noc = get_noc_addr(cxy >> 16, cxy & 0xFFFF, get_semaphore(done_sem_id));
    const uint64_t bc_h = get_noc_addr(bxy >> 16, bxy & 0xFFFF, bc_h_addr);
    const uint64_t gath_noc = get_noc_addr(bxy >> 16, bxy & 0xFFFF, get_semaphore(gath_sem_id));
    constexpr uint32_t peer0 = 11;
    volatile tt_l1_ptr uint32_t* sfree = sem(sfree_sem_id);
    volatile tt_l1_ptr uint32_t* hfree = sem(hfree_sem_id);
    const uint64_t pred_sfree = get_noc_addr(pxy >> 16, pxy & 0xFFFF, get_semaphore(sfree_sem_id));
    const uint64_t succ_xarr = get_noc_addr(sxy >> 16, sxy & 0xFFFF, get_semaphore(xarr_sem_id));
    const uint64_t succ_ring = get_noc_addr(sxy >> 16, sxy & 0xFFFF, x_ring);
    volatile tt_l1_ptr uint32_t* data_sem = sem(data_sem_id);
    volatile tt_l1_ptr uint32_t* done_sem = sem(done_sem_id);
    volatile tt_l1_ptr uint32_t* go_sem = sem(go_sem_id);
    volatile tt_l1_ptr uint32_t* xarr_sem = sem(xarr_sem_id);
    volatile tt_l1_ptr uint32_t* harr_sem[2] = {sem(harr_sem_id), sem(harr1_sem_id)};

    noc_semaphore_inc(credit_noc, w_slots);  // every landing slot starts free
    uint32_t granted = w_slots, pushed = 0;
    uint32_t x_pub = 0, x_cons = 0, x_fwd = 0, x_rep = 0;
    bool x_pending = false;
    uint32_t h_sent = 0, h_pending = 0, h_pub = 0;
    uint32_t out_done = 0, out_popped = 0, go_sent = 0;

    while (out_done < num_v) {
        invalidate_l1_cache();
        if (granted < total_w && cb_pages_reservable_at_back(in1_cb, (granted - pushed + 1) * slot)) {
            noc_semaphore_inc(credit_noc, 1);
            ++granted;
        }
        if (*data_sem > pushed) {
            cb_push_back(in1_cb, slot);
            ++pushed;
        }
        const uint32_t arrived = *xarr_sem;
        if (arrived > x_pub) {
            cb_push_back(x_cb, x_blk);
            ++x_pub;
        }
        if (x_cons < x_pub && cb_pages_reservable_at_back(x_cb, (x_slots - (x_pub - x_cons) + 1) * x_blk)) {
            ++x_cons;  // compute has consumed one more block
        }
        if (sxy && !x_pending && x_fwd < arrived && (x_fwd < x_slots || *sfree >= x_fwd + 1 - x_slots)) {
            const uint32_t off = (x_fwd % x_slots) * x_bytes;
            noc_async_write(x_ring + off, succ_ring + off, x_bytes);
            x_pending = true;
        }
        if (x_pending && ncrisc_noc_nonposted_writes_flushed(noc_index)) {
            noc_semaphore_inc(succ_xarr, 1);
            ++x_fwd;
            x_pending = false;
        }
        const uint32_t freed = sxy && x_fwd < x_cons ? x_fwd : x_cons;
        if (freed > x_rep) {
            x_rep = freed;
            if (pxy) {
                noc_inline_dw_write(pred_sfree, x_rep);
            } else {
                *hfree = x_rep;  // the chain head's DRAM reader
            }
        }
        if (!h_pending && h_sent < num_v && *go_sem + hbuf >= h_sent + 1 &&
            cb_pages_available_at_front(h_local_cb, mt * np)) {
            const uint32_t src = get_read_ptr(h_local_cb);
            const uint32_t dst = (h_sent % hbuf) * half_bytes;
            for (uint32_t m = 0; m < mt; ++m) {
                for (uint32_t p = 0; p < np; ++p) {  // h_all: row-major within K-blocks of KBLK tiles
                    const uint32_t k = cg * np + p;
                    const uint32_t off = ((k / kblk) * mt * kblk + m * kblk + k % kblk) * h_tile_bytes;
                    noc_async_write(src + (m * np + p) * h_tile_bytes, bc_h + dst + off, h_tile_bytes);
                }
            }
            h_pending = 1;
        }
        if (h_pending && ncrisc_noc_nonposted_writes_flushed(noc_index)) {
            noc_semaphore_inc(gath_noc, 1);
            cb_pop_front(h_local_cb, mt * np);
            ++h_sent;
            h_pending = 0;
        }
        if (h_pub < num_v && *harr_sem[h_pub % hbuf] >= h_pub / hbuf + 1 &&
            cb_pages_reservable_at_back(h_all_cb, h_all_tiles)) {
            cb_push_back(h_all_cb, h_all_tiles);
            ++h_pub;
        }
        if (cb_pages_available_at_front(out_cb, (out_done - out_popped + 1) * out_tiles)) {
            if (out_done + sub < num_v) {
                cb_pop_front(out_cb, out_tiles);  // the ring keeps the last S outputs (the last expert)
                ++out_popped;
            }
            noc_semaphore_inc(done_noc, 1);
            ++out_done;
        }
        if (is_coord && go_sent + 1 < num_v && *done_sem >= ncc * (go_sent + 1)) {
            for (uint32_t p = 0; p < ncc; ++p) {
                const uint32_t xy = get_arg_val<uint32_t>(peer0 + p);
                noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFF, get_semaphore(go_sem_id)), 1);
            }
            ++go_sent;
        }
    }
    noc_async_atomic_barrier();
}
