// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_typecast R0 `pure_copy`, page mode, writer half (BRISC / NoC1): drains the shared CB slot by slot and
// writes each batch of pages back (flush before releasing the slot, one final barrier).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "g6_typecast_copy_units.hpp"
#include "g6_typecast_zones.hpp"

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t cb = get_compile_time_arg_val(1);
    constexpr uint32_t buf_bytes = get_compile_time_arg_val(2);
    constexpr auto dst_args = TensorAccessorArgs<3>();

    const uint32_t dst_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t page0 = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    const uint32_t off = get_arg_val<uint32_t>(2);
    const uint32_t len = get_arg_val<uint32_t>(3);

    const auto dst = TensorAccessor(dst_args, dst_addr, page_bytes);
    g6_copy::for_each_batch(page0, count, off, len, buf_bytes, [&](uint32_t p, uint32_t n, uint32_t o, uint32_t b) {
        {
            G6_ZONE("copy_wait_data");
            cb_wait_front(cb, 1);
        }
        G6_ZONE("copy_write");
        const uint32_t l1 = get_read_ptr(cb);
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_write(l1 + i * b, dst.get_noc_addr(p + i, o), b);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb, 1);
    });
    G6_ZONE("copy_write_barrier");
    noc_async_write_barrier();
}
