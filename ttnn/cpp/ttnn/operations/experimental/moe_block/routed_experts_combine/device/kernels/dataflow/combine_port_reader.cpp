// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// combine_port_reader: a combine port's L1 hand-off receiver - grants the y writers injection-ring slots
// Runs on: combine port cores, only as the port role of combine_port_or_accumulator_reader.cpp (separate
//          accumulator cores; never a standalone kernel),
//          RISCV_1 (NCRISC), NoC 0 (combine dm_kernel: RISCV_1 is pinned to NoC 0)
// Responsibilities:
//  - reads this port's row count from the device plan (DRAM) over NoC 0; polls the writers' landed counters (L1)
//  - grants cb_inj ring slots to every flat-expert y writer (semaphore inc on their words over NoC 0)
//  - pushes a row into cb_inj once every writer landed it, for combine_port_writer to inject
//  - waits on the combine zero gate (rec::wait_released) before touching state

// Combine port reader: the L1 hand-off from the flat expert's y writers (y_row_major.hpp): the writers put every row
// of this port straight into its injection ring (cb_inj) in the order they produce them; this reader grants ring slots
// (grant[d] on every writer, cumulative), and pushes a row once every writer's landed counter here covers it (the
// column-0 writer also wrote the row's trailer: combine_formats.hpp). Its row count is the device plan's.
// The counters live in this core's MISC block (combine_formats.hpp: landed[] from kMiscPortLanded, the rows handed off
// before this call at kMiscPortHandoffBase - 0, as the state is zeroed every call).

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "combine_zero_gate.hpp"
#include "combine_formats.hpp"
#include "../combine_names.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/combine_plan_words.hpp"

// Compile-time args: all named (combine_names.hpp), the same whether this role runs alone or in a two-role kernel;
// the device plan's accessor args are the only positional ones
constexpr uint32_t cb_inj = get_named_compile_time_arg_val(combine_names::kInjectionCb);
constexpr uint32_t cb_tab = get_named_compile_time_arg_val(combine_names::kPortTableCb);
constexpr uint32_t MISC = get_named_compile_time_arg_val(combine_names::kMiscWordsOffset);
// the writers' words (arena-relative, on their cores)
constexpr uint32_t WORDS_OFF = get_named_compile_time_arg_val(combine_names::kYWriterWordsOffset);
constexpr uint32_t BAR_SLOT = get_named_compile_time_arg_val(combine_names::kBarrierCallSlot);
// (the device plan)
constexpr auto plan_args =
    TensorAccessorArgs<get_named_compile_time_arg_val(combine_names::kPortReaderPlanAccessorFirstArg)>();
constexpr uint32_t GRANT_B = 8;  // grants go out in batches of this many slots (or the rest)

#ifdef COMBINE_ROLE_FUNCTION_NAME  // (one role of a two-role kernel: combine_port_or_accumulator_reader / _writer.cpp)
void COMBINE_ROLE_FUNCTION_NAME() {
#else
void kernel_main() {
    moe_block_args::load();  // (a two-role kernel loads the core's args once, then picks this chip's role)
#endif
    if (moe_block_args::arg(4) == 0xFFFFFFFFu) {  // (an idle port core: another chip's port core - kIdlePort)
        return;
    }
    size_t a = 0;
    const uint32_t area = moe_block_args::arg(a++);  // the arena (same address on every core and chip)
    const uint32_t plan = moe_block_args::arg(a++);
    const uint32_t barrier_addr = moe_block_args::arg(a++);
    const uint32_t W = moe_block_args::arg(a++);
    const uint32_t port_id = moe_block_args::arg(a++), Q = moe_block_args::arg(a++);
    const uint32_t words = area + WORDS_OFF;
    const uint32_t call = rec::next_call(barrier_addr, BAR_SLOT);
    uint32_t n = 0;
    {
        cb_reserve_back(cb_tab, 1);
        const uint32_t scratch = (get_write_ptr(cb_tab) + 63) & ~63u;
        noc_async_read(TensorAccessor(plan_args, plan, moe_block_plan_words::kPlanBytes).get_noc_addr(0), scratch, 64);
        noc_async_read_barrier();
        n = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch)[port_id];
    }
    uint32_t writer_x[32], writer_y[32];
    for (uint32_t w = 0; w < W; ++w) {
        writer_x[w] = moe_block_args::arg(a++);
        writer_y[w] = moe_block_args::arg(a++);
    }
    rec::wait_released(barrier_addr, call);  // (this core's state and every writer's words zeroed)
    volatile tt_l1_ptr uint32_t* misc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + MISC);
    volatile tt_l1_ptr uint32_t* landed = misc + combine_formats::kMiscPortLanded;
    const uint32_t base = misc[combine_formats::kMiscPortHandoffBase];
    uint32_t granted = 0, pushed = 0;
    auto grant_to = [&](uint32_t g) {
        const uint32_t delta = g - granted;
        for (uint32_t w = 0; w < W; ++w) {
            noc_semaphore_inc(get_noc_addr(writer_x[w], writer_y[w], words + 4 * port_id), delta);
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
    misc[combine_formats::kMiscPortHandoffBase] = base + n;
    noc_async_atomic_barrier();
}
