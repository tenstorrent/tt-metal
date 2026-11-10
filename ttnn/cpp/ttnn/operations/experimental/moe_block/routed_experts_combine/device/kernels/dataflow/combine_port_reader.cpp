// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// combine_port_reader: a combine port's L1 hand-off receiver - grants the y writers injection-ring slots
// Runs on: combine port cells, only as combine_cell_reader's role 1 (ports-apart mode; never a standalone kernel),
//          RISCV_1 (NCRISC), NoC 0 (combine dm_kernel: RISCV_1 is pinned to NoC 0)
// Responsibilities:
//  - reads this port's row count from the device plan (DRAM) over NoC 0; polls the writers' landed counters (L1)
//  - grants cb_inj ring slots to every flat-expert y writer (semaphore inc on their words over NoC 0)
//  - pushes a row into cb_inj once every writer landed it, for combine_port_writer to inject
//  - waits on the combine zero gate (rec::wait_released) before touching state

// Combine port reader: the L1 hand-off from the flat expert's y writers (y_row_major.hpp SE_Y_L1): the writers put
// every row of this port straight into its injection ring (cb_inj) in the order they produce them; this reader grants
// ring slots (grant[d] on every writer, cumulative), and pushes a row once every writer's landed counter here covers it
// (the column-0 writer also wrote the trailer: header word and flags). Counters are cumulative within a call
// (misc[101]: this port's rows before this call; the state is zeroed every call). Its row count is the device plan's.

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "combine_zero_gate.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/combine_plan_words.hpp"

#ifndef CMB_CT0  // (the first compile-time arg of this role: 0 alone, past the other role's in a pool cell's binary)
#define CMB_CT0 0
#endif
constexpr uint32_t cb_inj = get_compile_time_arg_val(CMB_CT0 + 0);
constexpr uint32_t cb_tab = get_compile_time_arg_val(CMB_CT0 + 1);
constexpr uint32_t MISC = get_compile_time_arg_val(CMB_CT0 + 2);
constexpr uint32_t WORDS_OFF =
    get_compile_time_arg_val(CMB_CT0 + 3);  // the writers' words (arena-relative, on their cores)
constexpr uint32_t BAR_SLOT = get_compile_time_arg_val(CMB_CT0 + 4);
constexpr auto plan_args = TensorAccessorArgs<CMB_CT0 + CMB_PLAN_CT>();  // (the device plan)
constexpr uint32_t GRANT_B = 8;  // grants go out in batches of this many slots (or the rest)

#ifdef CMB_ROLE_FN  // (a role of a pool cell's binary - combine_cell_reader / _writer.cpp)
void CMB_ROLE_FN() {
#else
void kernel_main() {
#endif
#ifndef CMB_ROLE_FN
    moe_block_args::load();  // (a cell loads its block once, then picks its role: combine_cell_*.cpp)
#endif
    if (moe_block_args::arg(4) == 0xFFFFFFFFu) {  // (an idle port core: another chip's port cell - kIdlePort)
        return;
    }
    size_t a = 0;
    const uint32_t area = moe_block_args::arg(a++);  // the arena (same address on every core and chip)
    const uint32_t plan = moe_block_args::arg(a++);
    const uint32_t bar = moe_block_args::arg(a++);
    const uint32_t W = moe_block_args::arg(a++);
    const uint32_t did = moe_block_args::arg(a++), Q = moe_block_args::arg(a++);
    const uint32_t words = area + WORDS_OFF;
    const uint32_t call = rec::next_call(bar, BAR_SLOT);
    uint32_t n = 0;
    {
        cb_reserve_back(cb_tab, 1);
        const uint32_t sc = (get_write_ptr(cb_tab) + 63) & ~63u;
        noc_async_read(TensorAccessor(plan_args, plan, moe_block_plan_words::kPlanBytes).get_noc_addr(0), sc, 64);
        noc_async_read_barrier();
        n = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc)[did];
    }
    uint32_t wx[32], wy[32];
    for (uint32_t w = 0; w < W; ++w) {
        wx[w] = moe_block_args::arg(a++);
        wy[w] = moe_block_args::arg(a++);
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

#ifndef CMB_ROLE_FN
#endif
