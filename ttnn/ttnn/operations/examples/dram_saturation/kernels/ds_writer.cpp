// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// dram_saturation example writer (BRISC / NoC1).
//
// Drains the shared CB (filled by the reader) and writes this core's tiles back to
// interleaved DRAM, `block` writes per NoC barrier. No compute kernel — reader ->
// CB -> writer is a pure DRAM->DRAM copy. Byte-identical across variants and core
// counts; only the per-core (start_page, num_pages) runtime args change.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = 0;
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
            cb_wait_front(cb, b);
            const uint32_t l1_read_addr = get_read_ptr(cb);
            for (uint32_t i = 0; i < b; ++i) {
                noc_async_write(l1_read_addr + i * page_bytes, out_acc.get_noc_addr(start_page + p + i), page_bytes);
            }
            noc_async_write_barrier();
            cb_pop_front(cb, b);
            p += b;
        }
    }
}
