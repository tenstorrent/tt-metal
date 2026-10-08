// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Active-row untilize, writer (BRISC): each untilized block (32 rows x W * 32 columns bf16, c_16) into y_rm rows
// [32 tr, 32 tr + 32), column chunk c.
// CT: 0 NG, 1 EPC, 2 W, 3 NCH, 4 P
// Common RT: 0 y_rm addr, 1 counts, 2 regions, 3 lmap, 4 grid x
#include "untilize_active.hpp"
#include "core_range.hpp"

void kernel_main() {
    constexpr uint32_t NG = get_compile_time_arg_val(0), EPC = get_compile_time_arg_val(1);
    constexpr uint32_t W = get_compile_time_arg_val(2), NCH = get_compile_time_arg_val(3);
    constexpr uint32_t P = get_compile_time_arg_val(4);
    constexpr uint32_t SEG = W * 64, ROW = SEG * NCH;
    const InterleavedAddrGen<true> og = {.bank_base_address = get_common_arg_val<uint32_t>(0), .page_size = ROW};
    const uint32_t me = core_index(get_common_arg_val<uint32_t>(4));
    for_my_tile_rows<NG, EPC, NCH>(
        get_common_arg_val<uint32_t>(1),
        get_common_arg_val<uint32_t>(2),
        get_common_arg_val<uint32_t>(3),
        get_write_ptr(tt::CBIndex::c_5),
        me,
        P,
        [&](uint32_t tr, uint32_t c) {
            cb_wait_front(tt::CBIndex::c_16, W);
            const uint32_t src = get_read_ptr(tt::CBIndex::c_16);
            for (uint32_t r = 0; r < 32; ++r) {
                noc_async_write(src + r * SEG, get_noc_addr(tr * 32 + r, og, c * SEG), SEG);
            }
            noc_async_writes_flushed();
            cb_pop_front(tt::CBIndex::c_16, W);
        });
    noc_async_full_barrier();
}
