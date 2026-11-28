// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
// pure_copy_arms two-RISC variant, reader half: batches (copy_units.hpp) into CB slots of buf_bytes.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "copy_units.hpp"

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t cb = get_compile_time_arg_val(1);
    constexpr uint32_t buf_bytes = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t page0 = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    const uint32_t off = get_arg_val<uint32_t>(4);
    const uint32_t len = get_arg_val<uint32_t>(5);
    const auto src = TensorAccessor(src_args, src_addr, page_bytes);
    for_each_batch(page0, count, off, len, buf_bytes, [&](uint32_t p, uint32_t n, uint32_t o, uint32_t b) {
        cb_reserve_back(cb, 1);
        const uint32_t l1 = get_write_ptr(cb);
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_read(src.get_noc_addr(p + i, o), l1 + i * b, b);
        }
        noc_async_read_barrier();
        cb_push_back(cb, 1);
    });
}
