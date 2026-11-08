// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Microbenchmark (with gate_up_compute.cpp SE_BENCH): a RISC spinning like the flat expert's receiver / forwarder
// loops for CYCLES cycles - invalidate the L1 cache, poll CB counters / an L1 word, (MODE 1) read the NoC's flush
// status - with no data movement. RT: 0 CYCLES, 1 MODE, 2 the pause between polls (cycles, 0: none); CT: 0 X_CB, 1 W_CB
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t cycles = get_arg_val<uint32_t>(0), mode = get_arg_val<uint32_t>(1), pause = get_arg_val<uint32_t>(2);
    constexpr uint32_t x_cb = get_compile_time_arg_val(0), w_cb = get_compile_time_arg_val(1);
    volatile tt_reg_ptr uint32_t* clk = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    volatile tt_l1_ptr uint32_t* word = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    const uint32_t t0 = clk[0];
    uint32_t sink = 0;
    while (clk[0] - t0 < cycles) {
        invalidate_l1_cache();
        sink += *word;
        sink += cb_pages_reservable_at_back(x_cb, 1);
        sink += cb_pages_available_at_front(w_cb, 1);
        if (mode == 1) {
            sink += ncrisc_noc_nonposted_writes_flushed(noc_index);
        }
        if (pause) {
            const uint32_t t = clk[0];
            while (clk[0] - t < pause) {
            }
        }
    }
    *word = sink;  // (keep the polls)
}
