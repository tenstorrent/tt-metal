// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// x untilize into 2 KB pages, reader (NCRISC): blocks (tile row tr, 1024-column chunk c) j = me, me + P, ... of a
// TILE [rows, H] tensor (TILE_BYTES pages), 32 tiles each into c_0; the block count first (c_2 header, word 0).
// CT: 0 TILE_BYTES, 1 NCH (chunks per row = H / 1024), 2 P   Common RT: 0 x addr, 1 blocks, 2 grid x
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "core_range.hpp"

void kernel_main() {
    constexpr uint32_t TB = get_compile_time_arg_val(0), NCH = get_compile_time_arg_val(1);
    constexpr uint32_t P = get_compile_time_arg_val(2);
    const InterleavedAddrGen<true> xg = {.bank_base_address = get_common_arg_val<uint32_t>(0), .page_size = TB};
    const uint32_t blocks = get_common_arg_val<uint32_t>(1), me = core_index(get_common_arg_val<uint32_t>(2));
    cb_reserve_back(tt::CBIndex::c_2, 1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(tt::CBIndex::c_2)) =
        me < blocks ? (blocks - me + P - 1) / P : 0;
    cb_push_back(tt::CBIndex::c_2, 1);
    for (uint32_t j = me; j < blocks; j += P) {
        const uint32_t tr = j / NCH, c = j % NCH;
        cb_reserve_back(tt::CBIndex::c_0, 32);
        const uint32_t dst = get_write_ptr(tt::CBIndex::c_0);
        for (uint32_t i = 0; i < 32; ++i) {
            noc_async_read(get_noc_addr((tr * NCH + c) * 32 + i, xg), dst + i * TB, TB);
        }
        noc_async_read_barrier();
        cb_push_back(tt::CBIndex::c_0, 32);
    }
}
