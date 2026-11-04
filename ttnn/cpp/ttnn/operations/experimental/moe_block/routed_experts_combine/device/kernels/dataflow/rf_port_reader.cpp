// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Port reader, L1 hand-off from the flat expert's y writers (se_yrm.hpp SE_Y_L1): the writers put every row of this
// port straight into its injection ring (cb_inj) in the order they produce them; this reader grants ring slots
// (grant[d] on every writer, cumulative), and pushes a row once every writer's landed counter here covers it (the
// column-0 writer also wrote the trailer: header word and flags). Counters are cumulative within a call (misc[101]:
// this port's rows before this call; the state is zeroed every call). Its row count is the device plan's.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "combine_zero_gate.hpp"

constexpr uint32_t cb_inj = get_compile_time_arg_val(0);
constexpr uint32_t cb_tab = get_compile_time_arg_val(1);
constexpr uint32_t MISC = get_compile_time_arg_val(2);
constexpr uint32_t WORDS_OFF = get_compile_time_arg_val(3);  // the writers' words (arena-relative, on their cores)
constexpr uint32_t BAR_SLOT = get_compile_time_arg_val(4);
constexpr uint32_t GRANT_B = 8;  // grants go out in batches of this many slots (or the rest)

void kernel_main() {
    size_t a = 0;
    const uint32_t area = get_arg_val<uint32_t>(a++);  // the arena (same address on every core and chip)
    const uint32_t plan = get_arg_val<uint32_t>(a++);
    const uint32_t bar = get_arg_val<uint32_t>(a++);
    const uint32_t W = get_arg_val<uint32_t>(a++);
    const uint32_t did = get_arg_val<uint32_t>(a++), Q = get_arg_val<uint32_t>(a++);
    const uint32_t words = area + WORDS_OFF;
    const uint32_t call = rec::next_call(bar, BAR_SLOT);
    uint32_t n = 0;
    {
        cb_reserve_back(cb_tab, 1);
        const uint32_t sc = (get_write_ptr(cb_tab) + 63) & ~63u;
        const InterleavedAddrGen<true> pg = {.bank_base_address = plan, .page_size = 4096};
        noc_async_read(get_noc_addr(0, pg), sc, 64);
        noc_async_read_barrier();
        n = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc)[did];
    }
    uint32_t wx[32], wy[32];
    for (uint32_t w = 0; w < W; ++w) {
        wx[w] = get_arg_val<uint32_t>(a++);
        wy[w] = get_arg_val<uint32_t>(a++);
    }
    rec::wait_released(bar, call);  // (this core's state and every writer's words zeroed)
    volatile tt_l1_ptr uint32_t* misc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + MISC);
    volatile tt_l1_ptr uint32_t* landed = misc + 128;
    const uint32_t base = misc[101];
    uint32_t granted = 0, pushed = 0;
    auto grant_to = [&](uint32_t g) {
        const uint32_t dlt = g - granted;
        for (uint32_t w = 0; w < W; ++w) {
            noc_semaphore_inc(get_noc_addr(wx[w], wy[w], words + 4 * did), dlt);
        }
        granted = g;
    };
    while (pushed < n) {
        invalidate_l1_cache();
        const uint32_t want = pushed + Q < n ? pushed + Q : n;
        uint32_t g = granted;
        while (g < want && cb_pages_reservable_at_back(cb_inj, g - pushed + 1)) {
            ++g;
        }
        if (g > granted && (g - granted >= GRANT_B || g == want)) {
            grant_to(g);
        }
        uint32_t m = 0xFFFFFFFF;
        for (uint32_t w = 0; w < W; ++w) {
            const uint32_t v = landed[w] - base;
            m = v < m ? v : m;
        }
        while (pushed < m && pushed < granted) {
            cb_push_back(cb_inj, 1);
            ++pushed;
        }
    }
    misc[101] = base + n;
    noc_async_atomic_barrier();
}
