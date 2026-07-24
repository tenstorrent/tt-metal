// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// width_split example reader (NCRISC / NoC0).
//
// Reads this core's contiguous range of interleaved DRAM tiles into the
// reader->compute CB, `block` reads per NoC barrier. The kernel is BYTE-IDENTICAL
// for both variants; only the per-core (start_page, num_pages) runtime args differ:
//   single_core : one core gets ALL Wt tiles.
//   width_split : each core gets its WT_CHUNK-bounded tile-column chunk.
// So any measured difference is purely work distribution, not kernel code.

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
            noc_async_read_barrier();  // one barrier for `b` reads -> up to `block` in flight
            cb_push_back(cb_in, b);
            p += b;
        }
    }
}
