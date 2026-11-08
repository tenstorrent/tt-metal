// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Microbenchmark traffic (with gate_up_compute.cpp SE_BENCH): writes BYTES-byte pieces from this core's L1 into the
// successor core's L1, DEPTH in flight, paced to RATE bytes a 1024 cycles (0: unpaced), ITERS pieces - the gate/up
// cores' x snake (every core receives and forwards one stream). RT: 0 successor x, 1 successor y, 2 ITERS, 3 BYTES,
// 4 RATE; CT: 0 source CB, 1 destination CB
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t sx = get_arg_val<uint32_t>(0), sy = get_arg_val<uint32_t>(1);
    const uint32_t iters = get_arg_val<uint32_t>(2), bytes = get_arg_val<uint32_t>(3), rate = get_arg_val<uint32_t>(4);
    const uint32_t src = get_write_ptr(get_compile_time_arg_val(0));
    const uint64_t dst = get_noc_addr(sx, sy, get_write_ptr(get_compile_time_arg_val(1)));
    volatile tt_reg_ptr uint32_t* clk = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    const uint32_t per = rate ? (uint32_t)((uint64_t)bytes * 1024 / rate) : 0;  // cycles a piece
    uint32_t t = clk[0];
    for (uint32_t i = 0; i < iters; ++i) {
        if (per) {
            while ((int32_t)(clk[0] - t) < (int32_t)per) {
            }
            t += per;
        }
        noc_async_write(src, dst, bytes);
        if (i % 3 == 2) {
            noc_async_writes_flushed();
        }
    }
    noc_async_write_barrier();
}
