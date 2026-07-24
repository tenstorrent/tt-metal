// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// distribution_gate example writer (BRISC / NoC1).
//
// Drains the compute->writer CB and writes this core's (row_count x col_count)
// tile rectangle back to interleaved DRAM, `block` writes per NoC barrier. Tiles
// are walked in the SAME row-major order as the reader; page(r, c) = r * Wt + c.
// Byte-identical for every variant; only the per-core rectangle runtime args differ.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(1);
    constexpr uint32_t block = get_compile_time_arg_val(2);
    constexpr auto out_args = TensorAccessorArgs<3>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t row_count = get_arg_val<uint32_t>(2);
    const uint32_t col_start = get_arg_val<uint32_t>(3);
    const uint32_t col_count = get_arg_val<uint32_t>(4);
    const uint32_t Wt = get_arg_val<uint32_t>(5);

    const auto out_acc = TensorAccessor(out_args, dst_addr, page_bytes);
    const uint32_t total = row_count * col_count;

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        uint32_t i = 0;
        while (i < total) {
            const uint32_t b = (total - i) < block ? (total - i) : block;
            cb_wait_front(cb_out, b);
            const uint32_t l1_read_addr = get_read_ptr(cb_out);
            for (uint32_t j = 0; j < b; ++j) {
                const uint32_t idx = i + j;
                const uint32_t r = row_start + idx / col_count;
                const uint32_t c = col_start + idx % col_count;
                noc_async_write(l1_read_addr + j * page_bytes, out_acc.get_noc_addr(r * Wt + c), page_bytes);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, b);
            i += b;
        }
    }
}
