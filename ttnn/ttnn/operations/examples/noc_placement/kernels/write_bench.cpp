// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// noc_placement write-only micro-benchmark (mirror of read_bench.cpp).
//
// Isolates the DRAM WRITE stream so the NoC used for writes (NoC0 vs NoC1) can be
// compared with no read traffic to confound it. Writes this core's contiguous
// range of interleaved DRAM pages from a FIXED L1 scratch region of `block` pages,
// in BLOCKs (one barrier per block, so `block` writes in flight). No circular-buffer
// handshake and no reader, so nothing back-pressures the writer: the only thing
// measured is write bandwidth on the NoC selected by this kernel's config.
//
// The scratch content is meaningless (never read) -- write traffic is the point.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_scratch = 0;
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(1);
    constexpr uint32_t block = get_compile_time_arg_val(2);
    constexpr auto out_args = TensorAccessorArgs<3>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(1);
    const uint32_t num_pages = get_arg_val<uint32_t>(2);

    const auto out_acc = TensorAccessor(out_args, dst_addr, page_bytes);
    const uint32_t l1_scratch = get_read_ptr(cb_scratch);  // fixed L1 region, block pages

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        uint32_t p = 0;
        while (p < num_pages) {
            const uint32_t b = (num_pages - p) < block ? (num_pages - p) : block;
            for (uint32_t i = 0; i < b; ++i) {
                noc_async_write(l1_scratch + i * page_bytes, out_acc.get_noc_addr(start_page + p + i), page_bytes);
            }
            noc_async_write_barrier();  // one barrier per block -> `block` writes in flight
            p += b;
        }
    }
}
