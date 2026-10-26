// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// noc_placement copy writer (BRISC) — identical for every placement.
//
// Drains each page from the reader->writer CB and writes it back to the same
// interleaved page index (an identity copy). Runs on the NoC the reader does not
// use, so its writes overlap the reader's reads. Like the reader, this kernel never
// changes between placements — placement only moves the physical core that runs it.
//
// write_split (compile-time): 0 = every page is written on this kernel's NoC.
// k >= 1 = every k-th page is written on the OTHER NoC (the reader's), so 1/k of the
// write stream moves NoC; reads are untouched. Issuing on both NoCs from one RISC-V
// needs NOC_MODE::DM_DYNAMIC_NOC (the host sets it). The barrier then has to cover
// both NoCs before the CB pages are released. With write_split=0 the split code
// compiles out, so the baseline kernel is the plain single-NoC writer.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_id = 0;
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(1);
    constexpr uint32_t block = get_compile_time_arg_val(2);
    constexpr uint32_t write_split = get_compile_time_arg_val(3);
    constexpr auto out_args = TensorAccessorArgs<4>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(1);
    const uint32_t num_pages = get_arg_val<uint32_t>(2);

    const auto out_acc = TensorAccessor(out_args, dst_addr, page_bytes);
    constexpr uint8_t own_noc = noc_index;  // constexpr in kernel builds (the kernel's configured NoC)
    constexpr uint8_t other_noc = 1 - noc_index;

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        uint32_t p = 0;
        while (p < num_pages) {
            const uint32_t b = (num_pages - p) < block ? (num_pages - p) : block;
            cb_wait_front(cb_id, b);
            const uint32_t l1_read_addr = get_read_ptr(cb_id);
            for (uint32_t i = 0; i < b; ++i) {
                const uint32_t page = start_page + p + i;
                if constexpr (write_split == 0) {
                    noc_async_write(l1_read_addr + i * page_bytes, out_acc.get_noc_addr(page), page_bytes);
                } else if ((p + i) % write_split == write_split - 1) {
                    // every write_split-th page (by this core's local page count) takes the other NoC;
                    // each call names its NoC as a compile-time constant, so neither path pays for a
                    // runtime NoC index
                    noc_async_write(
                        l1_read_addr + i * page_bytes, out_acc.get_noc_addr(page, 0, other_noc), page_bytes, other_noc);
                } else {
                    noc_async_write(l1_read_addr + i * page_bytes, out_acc.get_noc_addr(page), page_bytes, own_noc);
                }
            }
            noc_async_write_barrier(own_noc);  // one barrier per block -> `block` writes in flight
            if constexpr (write_split != 0) {
                noc_async_write_barrier(other_noc);  // the pages moved to the other NoC must land too
            }
            cb_pop_front(cb_id, b);
            p += b;
        }
    }
}
