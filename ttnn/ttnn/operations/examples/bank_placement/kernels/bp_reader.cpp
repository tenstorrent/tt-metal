// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// bank_placement example reader (NCRISC / NoC0).
//
// Reads this core's pages `first, first + stride, first + 2*stride, ...` of an
// interleaved DRAM tensor into the CB, `block` reads per barrier. Page p lives in
// DRAM bank p % num_banks, so stride == num_banks means every read goes to ONE
// bank; stride == 1 walks all the banks. Byte-identical for every placement --
// only which core runs it changes.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = 0;
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t block = get_compile_time_arg_val(1);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(2);
    constexpr auto in_args = TensorAccessorArgs<3>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t stride = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);

    const auto acc = TensorAccessor(in_args, src_addr, page_bytes);

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        for (uint32_t i = 0; i < count; i += block) {
            const uint32_t b = (count - i) < block ? (count - i) : block;
            cb_reserve_back(cb, b);
            const uint32_t l1 = get_write_ptr(cb);
            for (uint32_t j = 0; j < b; ++j) {
                noc_async_read(acc.get_noc_addr(first + (i + j) * stride), l1 + j * page_bytes, page_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb, b);
        }
    }
}
