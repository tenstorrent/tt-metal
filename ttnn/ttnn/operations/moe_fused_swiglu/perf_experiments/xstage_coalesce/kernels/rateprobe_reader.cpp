// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED BAKE-OFF — raw per-transaction rate probe for the x-stick sub-page read pattern.
//
// Reader-only program (no compute, no writer): issue NUM_READS 1472B-class sub-page DRAM
// reads (the exact `x_acc.get_noc_addr(row, kstart_bytes)` pattern the real reader uses), ONE
// barrier, done. No tilize, no self-copy, no downstream consumer — this isolates the RAW
// read+barrier cost as a function of transaction COUNT, to fit against the ~110-125 ns/
// transaction "single-core transaction-rate-limited" ceiling `examples/double_buffer/report.md`
// measured for whole-tile DRAM reads, and see whether this op's SUB-PAGE interleaved reads sit
// on that same floor or far above it.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t X_SLICE = get_compile_time_arg_val(0);  // cb_x_in page: kr*32*2 bytes
constexpr uint32_t TA_BASE = 1;
constexpr auto x_args = TensorAccessorArgs<TA_BASE>();

constexpr uint32_t CB_X_IN = 0;
constexpr uint32_t TILE_H = 32;

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t kstart_bytes = get_arg_val<uint32_t>(1);
    const uint32_t kr = get_arg_val<uint32_t>(2);
    const uint32_t num_reads = get_arg_val<uint32_t>(3);  // how many of the (up to 32) sticks to read
    const uint32_t x_page = get_arg_val<uint32_t>(4);     // full stick bytes (emb*2) — the accessor's own page size

    const auto acc = TensorAccessor(x_args, x_addr, x_page);

    cb_reserve_back(CB_X_IN, TILE_H);
    const uint32_t wp = get_write_ptr(CB_X_IN);
    for (uint32_t s = 0; s < num_reads; ++s) {
        noc_async_read(acc.get_noc_addr(s, kstart_bytes), wp + s * X_SLICE, kr * TILE_H * 2);
    }
    noc_async_read_barrier();
    cb_push_back(CB_X_IN, TILE_H);
}
