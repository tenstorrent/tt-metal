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
    noc_semaphore_wait_min(sem, expect);
    noc_semaphore_set(sem, 0);

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
}
