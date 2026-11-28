// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t ca = get_compile_time_arg_val(0);
    constexpr uint32_t cb = get_compile_time_arg_val(1);
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t n = get_arg_val<uint32_t>(2);
    const uint32_t tb = get_tile_size(ca);
    for (uint32_t i = 0; i < n; ++i) {
        cb_reserve_back(ca, 1);
        cb_reserve_back(cb, 1);
        const uint32_t xy = get_arg_val<uint32_t>(3 + i);
        const uint32_t x = xy >> 16, y = (xy >> 8) & 0xFF, s = xy & 0xFF;
        noc_async_read(get_noc_addr(x, y, a_addr + s * tb), get_write_ptr(ca), tb);
        noc_async_read(get_noc_addr(x, y, b_addr + s * tb), get_write_ptr(cb), tb);
        noc_async_read_barrier();
        cb_push_back(ca, 1);
        cb_push_back(cb, 1);
    }
}
