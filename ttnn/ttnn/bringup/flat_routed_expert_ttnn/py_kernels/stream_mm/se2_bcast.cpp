// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Big-M streamed-expert broadcaster (BRISC, NOC0) on a core of its own, one per band of grid rows: it multicasts only
// into its band, so the bands' broadcasters never overlap (concurrent overlapping multicasts from several senders can
// deadlock the NoC) and x / h bandwidth scales with the number of bands. It serves both streams, h first:
//   * h: once all NCC slices of virtual expert v are in its h buffer (GATH), multicasts that half (v % HBUF) to the
//     compute cores' h_all at the same half, then multicast-sets HARR / HARR1 = v / HBUF + 1,
//   * x: multicasts staged x block g (from se2_xread.cpp) into the compute cores' x ring slot g % X_SLOTS once every
//     core of the band has freed that slot (XCRED, one credit per freed slot per core), then multicast-sets XARR = g
//     + 1.
// Multicasts go out in PIECE-byte pieces, one per loop pass, so a long x block never holds up a ready h.
//
// CT: 0 XS_CB, 1 X_BLK_TILES, 2 X_TILE_BYTES, 3 TOTAL_X_BLOCKS, 4 X_SLOTS, 5 NUM_V, 6 NCC, 7 HALF_BYTES, 8 HBUF,
//     9 GATH_SEM, 10 XCRED_SEM, 11 HARR_SEM, 12 HARR1_SEM, 13 XARR_SEM, 14 PIECE
// RT: 0 x ring address (compute cores), 1 h_all address (compute cores), 2 own h buffer address, 3 compute cores in the
//     band, 4 NUM_RECTS, then per rectangle: start xy, end xy (NOC0 order), destinations excluding self, then
//     NUM_H_RECTS and the h rectangles the same way (h may be served by one broadcaster for the whole grid: 0 = none)
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#ifdef SE_ZONES
#include "tools/profiler/kernel_profiler.hpp"
#define SE_MARK(name)            \
    {                            \
        DeviceZoneScopedN(name); \
    }
#else
#define SE_MARK(name)
#endif

void kernel_main() {
    constexpr uint32_t xs_cb = get_compile_time_arg_val(0);
    constexpr uint32_t x_blk = get_compile_time_arg_val(1);
    constexpr uint32_t x_tile = get_compile_time_arg_val(2);
    constexpr uint32_t total_x = get_compile_time_arg_val(3);
    constexpr uint32_t x_slots = get_compile_time_arg_val(4);
    constexpr uint32_t num_v = get_compile_time_arg_val(5);
    constexpr uint32_t ncc = get_compile_time_arg_val(6);
    constexpr uint32_t half_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t hbuf = get_compile_time_arg_val(8);
    constexpr uint32_t gath_sem_id = get_compile_time_arg_val(9);
    constexpr uint32_t xcred_sem_id = get_compile_time_arg_val(10);
    constexpr uint32_t harr_sem_id = get_compile_time_arg_val(11);
    constexpr uint32_t harr1_sem_id = get_compile_time_arg_val(12);
    constexpr uint32_t xarr_sem_id = get_compile_time_arg_val(13);
    constexpr uint32_t piece = get_compile_time_arg_val(14);
    constexpr uint32_t x_bytes = x_blk * x_tile;

    const uint32_t x_ring = get_arg_val<uint32_t>(0);
    const uint32_t h_all = get_arg_val<uint32_t>(1);
    const uint32_t own_h = get_arg_val<uint32_t>(2);
    const uint32_t band_ncc = get_arg_val<uint32_t>(3);
    const uint32_t num_rects = get_arg_val<uint32_t>(4);
    constexpr uint32_t max_rects = 4;
    uint64_t rect[max_rects];
    uint32_t dests[max_rects];
    for (uint32_t r = 0; r < num_rects; ++r) {
        const uint32_t a0 = get_arg_val<uint32_t>(5 + r * 3), a1 = get_arg_val<uint32_t>(6 + r * 3);
        rect[r] = get_noc_multicast_addr(a0 >> 16, a0 & 0xFFFF, a1 >> 16, a1 & 0xFFFF, 0);
        dests[r] = get_arg_val<uint32_t>(7 + r * 3);
    }
    const uint32_t h0 = 5 + num_rects * 3;
    const uint32_t num_h_rects = get_arg_val<uint32_t>(h0);
    uint64_t h_rect[max_rects];
    uint32_t h_dests[max_rects];
    for (uint32_t r = 0; r < num_h_rects; ++r) {
        const uint32_t a0 = get_arg_val<uint32_t>(h0 + 1 + r * 3), a1 = get_arg_val<uint32_t>(h0 + 2 + r * 3);
        h_rect[r] = get_noc_multicast_addr(a0 >> 16, a0 & 0xFFFF, a1 >> 16, a1 & 0xFFFF, 0);
        h_dests[r] = get_arg_val<uint32_t>(h0 + 3 + r * 3);
    }
    const uint32_t my_v = num_h_rects ? num_v : 0;
    auto sem = [](uint32_t id) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id)); };
    volatile tt_l1_ptr uint32_t* gath = sem(gath_sem_id);
    volatile tt_l1_ptr uint32_t* xcred = sem(xcred_sem_id);
    // Source words of the counter multicasts (in this core's copies of the counters, which nothing else uses here).
    volatile tt_l1_ptr uint32_t* harr_word[2] = {sem(harr_sem_id), sem(harr1_sem_id)};
    volatile tt_l1_ptr uint32_t* xarr_word = sem(xarr_sem_id);

    // Each rectangle gets its pieces in order; the last piece is linked to the counter multicast that follows it, so
    // the counter reaches every destination after the data without waiting for acks. A counter's source word is only
    // rewritten once everything issued before has left L1 (writes sent), so an earlier counter write can never pick up
    // a newer value.
    auto send = [&](uint32_t nr,
                    const uint64_t* rect,
                    const uint32_t* dests,
                    uint32_t src,
                    uint32_t dst,
                    uint32_t len,
                    bool last,
                    volatile tt_l1_ptr uint32_t* word,
                    uint32_t value) {
        if (last) {
            while (!ncrisc_noc_nonposted_writes_sent(noc_index)) {
            }
            *word = value;
        }
        for (uint32_t r = 0; r < nr; ++r) {
            if (dests[r]) {
                noc_async_write_multicast(src, rect[r] | dst, len, dests[r], last);
                if (last) {
                    noc_semaphore_set_multicast(
                        reinterpret_cast<uint32_t>(word), rect[r] | reinterpret_cast<uint32_t>(word), dests[r]);
                }
            }
        }
    };

    uint32_t hv = 0, h_off = 0;
    bool h_busy = false;
    uint32_t gx = 0, x_off = 0, x_src = 0;
    uint32_t x_state = 0;  // 0 idle, 1 sending pieces, 2 waiting for the block to leave L1 (then its slot is popped)
    while (hv < my_v || gx < total_x) {
        invalidate_l1_cache();
        if (!h_busy && hv < my_v && *gath >= ncc * (hv + 1)) {
            h_busy = true;
            h_off = 0;
        }
        if (x_state == 0 && gx < total_x && (gx < x_slots || *xcred >= band_ncc * (gx + 1 - x_slots)) &&
            cb_pages_available_at_front(xs_cb, x_blk)) {
            x_state = 1;
            x_off = 0;
            x_src = get_read_ptr(xs_cb);
            if (gx % 4 == 0) {
                SE_MARK("SB_X0");
            }
        }
        if (h_busy) {  // h first: down is waiting on it
            const uint32_t base = (hv % hbuf) * half_bytes + h_off;
            const uint32_t len = half_bytes - h_off < piece ? half_bytes - h_off : piece;
            h_off += len;
            const bool last = h_off == half_bytes;
            send(
                num_h_rects,
                h_rect,
                h_dests,
                own_h + base,
                h_all + base,
                len,
                last,
                harr_word[hv % hbuf],
                hv / hbuf + 1);
            if (last) {
                SE_MARK("SB_HSENT");
                ++hv;
                h_busy = false;
            }
        } else if (x_state == 1) {
            const uint32_t len = x_bytes - x_off < piece ? x_bytes - x_off : piece;
            const uint32_t off = x_off;
            x_off += len;
            const bool last = x_off == x_bytes;
            send(
                num_rects,
                rect,
                dests,
                x_src + off,
                x_ring + (gx % x_slots) * x_bytes + off,
                len,
                last,
                xarr_word,
                gx + 1);
            if (last) {
                x_state = 2;
                if (gx % 4 == 0) {
                    SE_MARK("SB_X1");
                }
            }
        }
        if (x_state == 2 && ncrisc_noc_nonposted_writes_sent(noc_index)) {
            cb_pop_front(xs_cb, x_blk);
            if (gx % 4 == 0) {
                SE_MARK("SB_XSENT");
            }
            ++gx;
            x_state = 0;
        }
    }
    noc_async_write_barrier();
}
