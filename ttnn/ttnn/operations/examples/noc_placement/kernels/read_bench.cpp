// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// noc_placement read-only micro-benchmark.
//
// Isolates the DRAM READ stream so the NoC used for reads (NoC0 vs NoC1) can be
// compared with no write traffic to confound it. Reads this core's contiguous
// range of interleaved DRAM pages, in BLOCKs (one barrier per block, so `block`
// reads in flight), into a FIXED L1 scratch region of `block` pages. There is no
// circular-buffer producer/consumer handshake and no writer, so nothing back-
// pressures the reader: the only thing being measured is read bandwidth on the
// NoC selected by this kernel's DataMovementConfigDescriptor.
//
// The scratch is reused every block (data is overwritten) — correctness is not
// the point here, read traffic is. `kernel_iters` repeats the whole range for a
// steady-state throughput measurement.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_scratch = 0;
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(1);
    constexpr uint32_t block = get_compile_time_arg_val(2);
    constexpr auto in_args = TensorAccessorArgs<3>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(1);
    const uint32_t num_pages = get_arg_val<uint32_t>(2);

    const auto in_acc = TensorAccessor(in_args, src_addr, page_bytes);
    const uint32_t l1_scratch = get_write_ptr(cb_scratch);  // fixed L1 region, block pages

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        uint32_t p = 0;
        while (p < num_pages) {
            const uint32_t b = (num_pages - p) < block ? (num_pages - p) : block;
            for (uint32_t i = 0; i < b; ++i) {
                noc_async_read(in_acc.get_noc_addr(start_page + p + i), l1_scratch + i * page_bytes, page_bytes);
            }
            noc_async_read_barrier();  // one barrier per block -> `block` reads in flight
            p += b;
        }
    }
}
