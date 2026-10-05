// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// bank_placement example writer (BRISC / NoC1).
//
// Writes the pages the reader fetched back to the SAME page indices of the output
// tensor, so each write goes to the same bank its read came from. Byte-identical
// for every placement.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = 0;
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t block = get_compile_time_arg_val(1);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(2);
    constexpr auto out_args = TensorAccessorArgs<3>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t stride = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);

    const auto acc = TensorAccessor(out_args, dst_addr, page_bytes);

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        for (uint32_t i = 0; i < count; i += block) {
            const uint32_t b = (count - i) < block ? (count - i) : block;
            cb_wait_front(cb, b);
            const uint32_t l1 = get_read_ptr(cb);
            for (uint32_t j = 0; j < b; ++j) {
                noc_async_write(l1 + j * page_bytes, acc.get_noc_addr(first + (i + j) * stride), page_bytes);
            }
            noc_async_writes_flushed();
            cb_pop_front(cb, b);
        }
    }
    noc_async_write_barrier();
}
