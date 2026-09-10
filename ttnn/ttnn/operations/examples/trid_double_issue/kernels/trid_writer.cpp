// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// trid_double_issue writer (BRISC / NoC1) — held CONSTANT across variants.
//
// Drains cb_in (filled by the reader on NoC0) straight back to the tensor's
// interleaved DRAM pages, so the op as a whole is an identity copy. There is no
// compute kernel: the CB goes reader -> writer directly, which keeps the math
// engines out of the measurement entirely. Writes ride NoC1 so they overlap the
// reader's NoC0 reads.
//
// This kernel is byte-identical for every variant, trid count and block size. It
// is not the thing being studied: holding the drain side fixed is what makes the
// measured delta attributable to the reader's barrier discipline.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_in = 0;
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
            cb_wait_front(cb_in, b);
            const uint32_t l1_read_addr = get_read_ptr(cb_in);
            for (uint32_t i = 0; i < b; ++i) {
                noc_async_write(l1_read_addr + i * page_bytes, out_acc.get_noc_addr(start_page + p + i), page_bytes);
            }
            // The slot only needs the writes to have left L1, not to have been
            // acked by the destination; the barrier at the end covers that.
            noc_async_writes_flushed();
            cb_pop_front(cb_in, b);
            p += b;
        }
    }
    noc_async_write_barrier();  // everything has actually landed before the kernel ends
}
