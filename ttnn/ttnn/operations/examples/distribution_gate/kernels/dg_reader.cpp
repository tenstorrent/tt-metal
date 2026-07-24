// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// distribution_gate example reader (NCRISC / NoC0).
//
// Reads this core's (row_count x col_count) rectangle of interleaved DRAM tiles
// into the reader->compute CB, `block` reads per NoC barrier. Tiles are walked in
// row-major order over the rectangle; page(r, c) = r * Wt + c. A full-width band
// (height split) is a contiguous page range; a column strip (width split) is a
// strided one — the same loop handles both. The kernel is BYTE-IDENTICAL for every
// variant; only the per-core rectangle runtime args differ, so any measured delta
// is purely work distribution, not kernel code.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(1);
    constexpr uint32_t block = get_compile_time_arg_val(2);
    constexpr auto in_args = TensorAccessorArgs<3>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t row_count = get_arg_val<uint32_t>(2);
    const uint32_t col_start = get_arg_val<uint32_t>(3);
    const uint32_t col_count = get_arg_val<uint32_t>(4);
    const uint32_t Wt = get_arg_val<uint32_t>(5);

    const auto in_acc = TensorAccessor(in_args, src_addr, page_bytes);
    const uint32_t total = row_count * col_count;

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        uint32_t i = 0;
        while (i < total) {
            const uint32_t b = (total - i) < block ? (total - i) : block;
            cb_reserve_back(cb_in, b);
            const uint32_t l1_write_addr = get_write_ptr(cb_in);
            for (uint32_t j = 0; j < b; ++j) {
                const uint32_t idx = i + j;
                const uint32_t r = row_start + idx / col_count;
                const uint32_t c = col_start + idx % col_count;
                noc_async_read(in_acc.get_noc_addr(r * Wt + c), l1_write_addr + j * page_bytes, page_bytes);
            }
            noc_async_read_barrier();  // one barrier for `b` reads -> up to `block` in flight
            cb_push_back(cb_in, b);
            i += b;
        }
    }
}
