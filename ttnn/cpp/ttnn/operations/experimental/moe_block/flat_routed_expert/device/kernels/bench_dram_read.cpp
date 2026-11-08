// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Microbenchmark DRAM load (with gate_up_compute.cpp SE_BENCH): reads pages of an interleaved DRAM tensor into a
// scratch CB for CYCLES cycles, BATCH reads a barrier - the chip's DRAM saturated while the compute runs.
// RT: 0 tensor address, 1 pages, 2 CYCLES, 3 first page; CT: 0 scratch CB, 1 BATCH, 2.. the tensor's accessor
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0), batch = get_compile_time_arg_val(1);
    constexpr auto ta = TensorAccessorArgs<2>();
    const uint32_t pages = get_arg_val<uint32_t>(1), cycles = get_arg_val<uint32_t>(2);
    uint32_t pg = get_arg_val<uint32_t>(3);
    const uint32_t psize = get_tile_size(cb);
    const auto t = TensorAccessor(ta, get_arg_val<uint32_t>(0), psize);
    volatile tt_reg_ptr uint32_t* clk = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    const uint32_t t0 = clk[0], dst = get_write_ptr(cb);
    while (clk[0] - t0 < cycles) {
        for (uint32_t i = 0; i < batch; ++i) {
            noc_async_read_tile(pg, t, dst + (i % 4) * psize);
            pg = pg + 1 == pages ? 0 : pg + 1;
        }
        noc_async_read_barrier();
    }
}
