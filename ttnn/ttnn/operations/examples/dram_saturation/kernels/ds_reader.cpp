// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// dram_saturation example reader (NCRISC / NoC0).
//
// Reads this core's contiguous range of interleaved DRAM tiles into the shared CB,
// `block` reads per NoC barrier. There is no compute kernel — the writer drains the
// same CB and writes back to DRAM, so this is a pure DRAM->DRAM copy (read on NoC0,
// write on NoC1). The kernel is BYTE-IDENTICAL across variants and core counts;
// only the per-core (start_page, num_pages) runtime args and how many cores run
// change, so any measured bandwidth delta is purely work distribution + placement.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = 0;
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
            cb_reserve_back(cb, b);
            const uint32_t l1_write_addr = get_write_ptr(cb);
            for (uint32_t i = 0; i < b; ++i) {
                noc_async_read(in_acc.get_noc_addr(start_page + p + i), l1_write_addr + i * page_bytes, page_bytes);
            }
            noc_async_read_barrier();  // one barrier per `block` reads -> up to `block` in flight
            cb_push_back(cb, b);
            p += b;
        }
    }
}
