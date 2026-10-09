// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Active-row untilize, reader (NCRISC): this core's active tile rows of y (bfp8 TILE [rows, H]), W tiles at a time
// (c_0); the block count first (c_2 header, word 0).
// CT: 0 NG, 1 EPC, 2 TILE_BYTES, 3 W (tiles per block), 4 NCH (blocks per tile row), 5 P
// Common RT: 0 y addr, 1 counts, 2 regions, 3 lmap, 4 grid x
#include "untilize_active.hpp"
#include "core_range.hpp"

void kernel_main() {
    constexpr uint32_t NG = get_compile_time_arg_val(0), EPC = get_compile_time_arg_val(1);
    constexpr uint32_t TB = get_compile_time_arg_val(2), W = get_compile_time_arg_val(3);
    constexpr uint32_t NCH = get_compile_time_arg_val(4), P = get_compile_time_arg_val(5);
    const InterleavedAddrGen<true> yg = {.bank_base_address = get_common_arg_val<uint32_t>(0), .page_size = TB};
    const uint32_t me = core_index(get_common_arg_val<uint32_t>(4));
    const uint32_t l1 = get_write_ptr(tt::CBIndex::c_4);
    uint32_t n = 0;
    for_my_tile_rows<NG, EPC, NCH>(
        get_common_arg_val<uint32_t>(1),
        get_common_arg_val<uint32_t>(2),
        get_common_arg_val<uint32_t>(3),
        l1,
        me,
        P,
        [&](uint32_t, uint32_t) { ++n; });
    cb_reserve_back(tt::CBIndex::c_2, 1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(tt::CBIndex::c_2)) = n;
    cb_push_back(tt::CBIndex::c_2, 1);
    for_my_tile_rows<NG, EPC, NCH>(
        get_common_arg_val<uint32_t>(1),
        get_common_arg_val<uint32_t>(2),
        get_common_arg_val<uint32_t>(3),
        l1,
        me,
        P,
        [&](uint32_t tr, uint32_t c) {
            cb_reserve_back(tt::CBIndex::c_0, W);
            const uint32_t dst = get_write_ptr(tt::CBIndex::c_0);
            for (uint32_t i = 0; i < W; ++i) {
                noc_async_read(get_noc_addr((tr * NCH + c) * W + i, yg), dst + i * TB, TB);
            }
            noc_async_read_barrier();
            cb_push_back(tt::CBIndex::c_0, W);
        });
}
