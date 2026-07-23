// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// index_staging writer (BRISC / NoC1).
//
// Drains each finished, selected output row from the reader->writer CB (cb_out)
// and writes it back to its DRAM page. Runs on NoC1 so its writes overlap the
// reader's NoC0 reads. It is byte-identical for both access-strategy variants —
// the strategy lives entirely in the reader (see is_reader.cpp) — so the writer
// never contributes to the measured delta.

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_out = 16;

void kernel_main() {
    constexpr uint32_t out_page_bytes = get_compile_time_arg_val(0);  // whole output row
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(1);
    constexpr auto out_args = TensorAccessorArgs<2>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(2);

    const auto out_acc = TensorAccessor(out_args, dst_addr, out_page_bytes);

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        for (uint32_t r = 0; r < num_rows; ++r) {
            const uint32_t row = start_row + r;
            cb_wait_front(cb_out, 1);
            noc_async_write(get_read_ptr(cb_out), out_acc.get_noc_addr(row), out_page_bytes);
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}
