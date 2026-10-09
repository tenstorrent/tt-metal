// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// M-split big-M relay (BRISC, NOC0) on a core of its own: the program's only multicast sender. Each of the G M-groups
// (one per run of physically contiguous columns, one rectangle) only needs its own rows, so every byte of x and h is
// multicast exactly once, into one rectangle:
//   * h (first: down is waiting on it): once all cores of group g have put their slices of virtual expert v into the
//     group's buffer here (group GATH), multicasts it to the group's h_all and sets its HARR = v + 1 (single buffer),
//   * x: staged blocks (se2_xread.cpp) come in the order (v, K-block, group); block n of group g goes to x ring slot
//     n % X_SLOTS once every core of the group has freed that slot (the minimum over the group's x-freed words, which
//     the cores write here), then the group's XARR = n + 1.
// The last data piece of a block is linked to the counter multicast that follows it, so the counter lands after the
// data without waiting for acks; a counter's source word is only rewritten once everything before has left L1.
//
// CT: 0 XS_CB, 1 X_BLK_TILES, 2 X_TILE_BYTES, 3 TOTAL_X_BLOCKS (all groups), 4 X_SLOTS, 5 NUM_V, 6 GROUP_NCC,
//     7 HALF_BYTES (one group's h), 8 PIECE, 9 HARR_SEM, 10 XARR_SEM, 11 G
// RT: 0 x ring address, 1 h_all address, 2 own h buffer base, then per group: start xy, end xy, destinations excluding
//     self, GATH sem id, address of the group's GROUP_NCC x-freed words, X source-word sem id, H source-word sem id
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t xs_cb = get_compile_time_arg_val(0);
    constexpr uint32_t x_blk = get_compile_time_arg_val(1);
    constexpr uint32_t x_tile = get_compile_time_arg_val(2);
    constexpr uint32_t total_x = get_compile_time_arg_val(3);
    constexpr uint32_t x_slots = get_compile_time_arg_val(4);
    constexpr uint32_t num_v = get_compile_time_arg_val(5);
    constexpr uint32_t group_ncc = get_compile_time_arg_val(6);
    constexpr uint32_t half_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t piece = get_compile_time_arg_val(8);
    constexpr uint32_t harr_sem_id = get_compile_time_arg_val(9);
    constexpr uint32_t xarr_sem_id = get_compile_time_arg_val(10);
    constexpr uint32_t groups = get_compile_time_arg_val(11);
    constexpr uint32_t x_bytes = x_blk * x_tile;
    static_assert(groups <= 4);

    const uint32_t x_ring = get_arg_val<uint32_t>(0);
    const uint32_t h_all = get_arg_val<uint32_t>(1);
    const uint32_t own_h = get_arg_val<uint32_t>(2);
    auto sem = [](uint32_t id) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id)); };
    uint64_t rect[groups];
    uint32_t dests[groups];
    volatile tt_l1_ptr uint32_t* gath[groups];
    volatile tt_l1_ptr uint32_t* xfreed[groups];
    volatile tt_l1_ptr uint32_t* xword[groups];
    volatile tt_l1_ptr uint32_t* hword[groups];
    for (uint32_t g = 0; g < groups; ++g) {
        const uint32_t a = 3 + g * 7;
        const uint32_t a0 = get_arg_val<uint32_t>(a), a1 = get_arg_val<uint32_t>(a + 1);
        rect[g] = get_noc_multicast_addr(a0 >> 16, a0 & 0xFFFF, a1 >> 16, a1 & 0xFFFF, 0);
        dests[g] = get_arg_val<uint32_t>(a + 2);
        gath[g] = sem(get_arg_val<uint32_t>(a + 3));
        xfreed[g] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(a + 4));
        xword[g] = sem(get_arg_val<uint32_t>(a + 5));
        hword[g] = sem(get_arg_val<uint32_t>(a + 6));
    }
    const uint32_t harr_addr = get_semaphore(harr_sem_id), xarr_addr = get_semaphore(xarr_sem_id);

    auto min_freed = [&](uint32_t g) {
        uint32_t lo = xfreed[g][0];
        for (uint32_t i = 1; i < group_ncc; ++i) {
            lo = xfreed[g][i] < lo ? xfreed[g][i] : lo;
        }
        return lo;
    };
    auto send = [&](uint32_t g,
                    uint32_t src,
                    uint32_t dst,
                    uint32_t len,
                    bool last,
                    volatile tt_l1_ptr uint32_t* word,
                    uint32_t cnt_addr,
                    uint32_t value) {
#ifdef SE_NO_LINK
        // Debug variant: the counter goes out only after the data is acknowledged.
        noc_async_write_multicast(src, rect[g] | dst, len, dests[g]);
        if (last) {
            noc_async_write_barrier();
            *word = value;
            noc_semaphore_set_multicast(reinterpret_cast<uint32_t>(word), rect[g] | cnt_addr, dests[g]);
            noc_async_write_barrier();
        }
#else
        if (last) {
            while (!ncrisc_noc_nonposted_writes_sent(noc_index)) {
            }
            *word = value;
        }
        noc_async_write_multicast(src, rect[g] | dst, len, dests[g], last);
        if (last) {
            noc_semaphore_set_multicast(reinterpret_cast<uint32_t>(word), rect[g] | cnt_addr, dests[g]);
        }
#endif
    };

    uint32_t hv[groups], h_off[groups];
    bool h_busy[groups];
    for (uint32_t g = 0; g < groups; ++g) {
        hv[g] = 0;
        h_off[g] = 0;
        h_busy[g] = false;
    }
    uint32_t gx = 0, x_off = 0, x_src = 0, x_state = 0;  // 0 idle, 1 sending, 2 waiting to leave L1
    uint32_t h_left = groups * num_v;
    while (h_left || gx < total_x) {
        invalidate_l1_cache();
        bool sent_h = false;
        for (uint32_t g = 0; g < groups && !sent_h; ++g) {
            if (!h_busy[g] && hv[g] < num_v && *gath[g] >= group_ncc * (hv[g] + 1)) {
                h_busy[g] = true;
                h_off[g] = 0;
            }
            if (h_busy[g]) {
                const uint32_t len = half_bytes - h_off[g] < piece ? half_bytes - h_off[g] : piece;
                const uint32_t off = h_off[g];
                h_off[g] += len;
                const bool last = h_off[g] == half_bytes;
                send(g, own_h + g * half_bytes + off, h_all + off, len, last, hword[g], harr_addr, hv[g] + 1);
                if (last) {
                    ++hv[g];
                    --h_left;
                    h_busy[g] = false;
                }
                sent_h = true;
            }
        }
        const uint32_t xg = gx % groups, xn = gx / groups;
        if (x_state == 0 && gx < total_x && (xn < x_slots || min_freed(xg) >= xn + 1 - x_slots) &&
            cb_pages_available_at_front(xs_cb, x_blk)) {
            x_state = 1;
            x_off = 0;
            x_src = get_read_ptr(xs_cb);
        }
        if (!sent_h && x_state == 1) {
            const uint32_t len = x_bytes - x_off < piece ? x_bytes - x_off : piece;
            const uint32_t off = x_off;
            x_off += len;
            const bool last = x_off == x_bytes;
            send(xg, x_src + off, x_ring + (xn % x_slots) * x_bytes + off, len, last, xword[xg], xarr_addr, xn + 1);
            if (last) {
                x_state = 2;
            }
        }
        if (x_state == 2 && ncrisc_noc_nonposted_writes_sent(noc_index)) {
            cb_pop_front(xs_cb, x_blk);
            ++gx;
            x_state = 0;
        }
    }
    noc_async_write_barrier();
}
