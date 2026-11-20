// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// mhc_post feed port reader (deepseek_v41_flash_d_p P.36; NCRISC, NoC 0). Port (side d, link l): its targets are my
// group's positions t != me with side(me, t) == d (the feed table, common args 1..). Waits until every unit it
// carries has staged its tiles in y (one SEM_FEED increment per (unit, target) from the unit's writer), then reads its
// streams (mhc_post_feed_common.hpp), target by target, into packet slots of cb_feed (up to P tiles a slot) for the
// sender. Runtime args: d, l; common arg 0: y.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "mhc_post_feed_common.hpp"
#include "perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_feed = get_compile_time_arg_val(0);
    constexpr uint32_t page = get_compile_time_arg_val(1);
    constexpr uint32_t P = get_compile_time_arg_val(2);
    constexpr uint32_t rt = get_compile_time_arg_val(3);
    constexpr uint32_t bw = get_compile_time_arg_val(4);
    constexpr uint32_t ct = get_compile_time_arg_val(5);
    constexpr uint32_t cpt = get_compile_time_arg_val(6);
    constexpr uint32_t sem_feed_id = get_compile_time_arg_val(7);
    constexpr auto y_args = TensorAccessorArgs<8>();
    const auto y_acc = TensorAccessor(y_args, get_common_arg_val<uint32_t>(0), page);
    const mhc_feed::Table<1> ft;
    const uint32_t links = ft.L;

    const uint32_t d = get_arg_val<uint32_t>(0);
    const uint32_t l = get_arg_val<uint32_t>(1);
    uint32_t expect = 0;
    for (uint32_t t = 0; t < ft.G; ++t) {
        if (t != ft.me && ft.side(ft.me, t) == d) {
            for (uint32_t u = ft.u_lo(t) + l; u < ft.u_hi(t); u += links) {
                ++expect;
            }
        }
    }
    if (expect == 0) {
        return;
    }
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_feed_id));
    {
        MaybeDeviceZoneScope("fd_wait");
        noc_semaphore_wait_min(sem, expect);
    }
    noc_semaphore_set(sem, 0);
    MaybeDeviceZoneScope("fd_read");

#ifdef MHC_POST_FAST
    // P.83: up to DEPTH packets' reads in flight (transaction id 1 + packet mod DEPTH), pushed in order as each lands
    // (a barrier a packet left the port's DRAM re-read latency-bound: 22-50 us for its 0.3-0.6 MB on the 8x4). DEPTH =
    // the slots the sender never holds: it keeps up to 4 (its packet headers) unpopped until a flush, so reserving
    // more than SLOTS - 4 ahead deadlocks against it.
    constexpr uint32_t SLOTS = MHC_POST_FEED_SLOTS;  // cb_feed's packet slots (FEED_SLOTS_FAST)
    constexpr uint32_t DEPTH = SLOTS - 4;
    const uint32_t cb_base = get_write_ptr(cb_feed);  // (nothing pushed yet: slot 0)
    uint32_t issued = 0, pushed = 0;
    auto push_one = [&]() {
        noc_async_read_barrier_with_trid(1 + pushed % DEPTH);  // (ids 1..DEPTH: the packets in flight)
        cb_push_back(cb_feed, 1);
        ++pushed;
    };
    for (uint32_t t = 0; t < ft.G; ++t) {
        if (t == ft.me || ft.side(ft.me, t) != d) {
            continue;
        }
        const uint32_t u_lo = ft.u_lo(t), u_hi = ft.u_hi(t);
        uint32_t left = mhc_feed::count_tiles(t, u_lo, u_hi, l, links, rt, bw, ct, cpt);
        uint32_t in_pkt = 0, size = 0, dst = 0;
        mhc_feed::for_each_tile(t, u_lo, u_hi, l, links, rt, bw, ct, cpt, [&](uint32_t r, uint32_t k) {
            if (in_pkt == 0) {
                if (issued - pushed == DEPTH) {
                    push_one();
                }
                size = mhc_feed::packet_size(left, P);
                cb_reserve_back(cb_feed, issued - pushed + 1);
                dst = cb_base + (issued % SLOTS) * P * page;
                noc_async_read_set_trid(1 + issued % DEPTH);
            }
            noc_async_read_page(r * ct + k, y_acc, dst + in_pkt * page);
            --left;
            if (++in_pkt == size) {
                ++issued;
                in_pkt = 0;
            }
        });
    }
    noc_async_read_set_trid(0);
    while (pushed < issued) {
        push_one();
    }
#else
    for (uint32_t t = 0; t < ft.G; ++t) {
        if (t == ft.me || ft.side(ft.me, t) != d) {
            continue;
        }
        const uint32_t u_lo = ft.u_lo(t), u_hi = ft.u_hi(t);
        uint32_t left = mhc_feed::count_tiles(t, u_lo, u_hi, l, links, rt, bw, ct, cpt);
        uint32_t in_pkt = 0, size = 0, dst = 0;
        mhc_feed::for_each_tile(t, u_lo, u_hi, l, links, rt, bw, ct, cpt, [&](uint32_t r, uint32_t k) {
            if (in_pkt == 0) {
                size = mhc_feed::packet_size(left, P);
                cb_reserve_back(cb_feed, 1);
                dst = get_write_ptr(cb_feed);
            }
            noc_async_read_page(r * ct + k, y_acc, dst + in_pkt * page);
            --left;
            if (++in_pkt == size) {
                noc_async_read_barrier();
                cb_push_back(cb_feed, 1);
                in_pkt = 0;
            }
        });
    }
#endif
}
