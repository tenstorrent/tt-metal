// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// noc_placement copy writer (BRISC) — identical for every placement.
//
// Drains each page from the reader->writer CB and writes it back to the same
// interleaved DRAM page index (an identity copy). Runs on NoC1 so its writes
// overlap the reader's NoC0 reads. Like the reader, this kernel never changes
// between placements — placement only moves the physical core that runs it.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_id = 0;
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(1);
    constexpr uint32_t block = get_compile_time_arg_val(2);
    constexpr auto out_args = TensorAccessorArgs<3>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(1);
    const uint32_t num_pages = get_arg_val<uint32_t>(2);

    const auto out_acc = TensorAccessor(out_args, dst_addr, page_bytes);

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        uint32_t p = 0;
        while (p < num_pages) {
            const uint32_t b = (num_pages - p) < block ? (num_pages - p) : block;
            cb_wait_front(cb_id, b);
            const uint32_t l1_read_addr = get_read_ptr(cb_id);
            for (uint32_t i = 0; i < b; ++i) {
                noc_async_write(l1_read_addr + i * page_bytes, out_acc.get_noc_addr(start_page + p + i), page_bytes);
            }
            noc_async_write_barrier();  // one barrier per block -> `block` writes in flight
            cb_pop_front(cb_id, b);
            p += b;
        }
    }
}
