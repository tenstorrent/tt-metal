// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t co = get_compile_time_arg_val(0);
    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    const uint32_t tb = get_tile_size(co);
    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(co, 1);
        const uint32_t xy = get_arg_val<uint32_t>(2 + i);
        const uint32_t x = xy >> 16, y = (xy >> 8) & 0xFF, s = xy & 0xFF;
        noc_async_write(get_read_ptr(co), get_noc_addr(x, y, o_addr + s * tb), tb);
        noc_async_write_barrier();
        cb_pop_front(co, 1);
    }
}
