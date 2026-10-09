// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Row add with a tiled output, writer (BRISC): each block's 32 tiles (c_16) at tile row tr, tile columns 32 c ...
// CT: 0 NCH, 1 P, 2 TILES_PER_ROW
// Common RT: 0 out addr, 1 blocks, 2 grid x
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "core_range.hpp"

void kernel_main() {
    constexpr uint32_t NCH = get_compile_time_arg_val(0), P = get_compile_time_arg_val(1);
    constexpr uint32_t TPR = get_compile_time_arg_val(2);
    const InterleavedAddrGen<true> o = {.bank_base_address = get_common_arg_val<uint32_t>(0), .page_size = 2048};
    const uint32_t blocks = get_common_arg_val<uint32_t>(1), me = core_index(get_common_arg_val<uint32_t>(2));
    for (uint32_t j = me; j < blocks; j += P) {
        const uint32_t tr = j / NCH, c = j % NCH;
        cb_wait_front(tt::CBIndex::c_16, 32);
        const uint32_t src = get_read_ptr(tt::CBIndex::c_16);
        for (uint32_t i = 0; i < 32; ++i) {
            noc_async_write(src + i * 2048, get_noc_addr(tr * TPR + c * 32 + i, o), 2048);
        }
        noc_async_writes_flushed();
        cb_pop_front(tt::CBIndex::c_16, 32);
    }
    noc_async_full_barrier();
}
