// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// noc_placement copy reader (NCRISC) — identical for every placement.
//
// Reads this core's contiguous range of interleaved DRAM pages, one whole page
// at a time, into the reader->writer CB. The kernel is byte-for-byte the same
// no matter where the core sits on the grid; the ONLY thing the example varies
// between runs is WHICH physical cores run this kernel (a row, a column, or a
// diagonal). So any measured difference is attributable purely to the NoC paths
// those placements take to reach DRAM — not to the work done.
//
// kernel_iters loops the whole read range in-kernel: iters=1 measures per-launch
// latency, large iters measures steady-state read throughput under contention.
//
// Reads are issued in BLOCKs with a single barrier per block, so up to `block`
// reads are in flight at once. That is deliberate: one-read-then-barrier is
// LATENCY-bound (it measures distance to DRAM), whereas many outstanding reads
// saturate the NoC and expose LINK CONTENTION -- which is the thing placement
// actually changes.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(1);
    constexpr uint32_t block = get_compile_time_arg_val(2);
    constexpr auto in_args = TensorAccessorArgs<3>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(1);
    const uint32_t num_pages = get_arg_val<uint32_t>(2);

    const auto in_acc = TensorAccessor(in_args, src_addr, page_bytes);

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        uint32_t p = 0;
        while (p < num_pages) {
            const uint32_t b = (num_pages - p) < block ? (num_pages - p) : block;
            cb_reserve_back(cb_in, b);
            const uint32_t l1_write_addr = get_write_ptr(cb_in);
            for (uint32_t i = 0; i < b; ++i) {
                noc_async_read(in_acc.get_noc_addr(start_page + p + i), l1_write_addr + i * page_bytes, page_bytes);
            }
            noc_async_read_barrier();  // one barrier per block -> `block` reads in flight
            cb_push_back(cb_in, b);
            p += b;
        }
    }
}
