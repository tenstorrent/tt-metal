// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Row add (out[i] = a[a_off + i] + b[b_off + i], row-major bf16), reader (NCRISC): rows [r0, r0 + n) of a and b as
// TILES tiles of 1024 elements into c_0 / c_1. b_off = the per-device chip-info word 1 when INFO (the exchange: the
// peer's block of the gathered partials), else RT 5; a_off = RT 6.
// CT: 0 ROW_BYTES, 1 TILES, 2 INFO, 3 BATCH (rows per read barrier)
// Common RT: 0 a addr, 1 b addr, 2 chip-info addr, 3 rows, 4 rows per core (range r0, n), 5 b_off, 6 a_off,
//     7 grid x
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "core_range.hpp"

void kernel_main() {
    constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(0);
    constexpr uint32_t TILES = get_compile_time_arg_val(1);
    constexpr bool INFO = get_compile_time_arg_val(2) != 0;
    constexpr uint32_t BATCH = get_compile_time_arg_val(3);
    const InterleavedAddrGen<true> a = {.bank_base_address = get_common_arg_val<uint32_t>(0), .page_size = ROW_BYTES};
    const InterleavedAddrGen<true> b = {.bank_base_address = get_common_arg_val<uint32_t>(1), .page_size = ROW_BYTES};
    const auto [r0, n] =
        core_range(get_common_arg_val<uint32_t>(3), get_common_arg_val<uint32_t>(4), get_common_arg_val<uint32_t>(7));
    const uint32_t a_off = get_common_arg_val<uint32_t>(6);
    uint32_t b_off = get_common_arg_val<uint32_t>(5);
    if constexpr (INFO) {
        const uint32_t l1 = get_write_ptr(tt::CBIndex::c_7);
        noc_async_read(
            get_noc_addr(
                0, InterleavedAddrGen<true>{.bank_base_address = get_common_arg_val<uint32_t>(2), .page_size = 64}),
            l1,
            64);
        noc_async_read_barrier();
        b_off = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1)[1];
    }
    for (uint32_t i = 0; i < n; i += BATCH) {
        const uint32_t m = n - i < BATCH ? n - i : BATCH;
        cb_reserve_back(tt::CBIndex::c_0, m * TILES);
        cb_reserve_back(tt::CBIndex::c_1, m * TILES);
        const uint32_t la = get_write_ptr(tt::CBIndex::c_0), lb = get_write_ptr(tt::CBIndex::c_1);
        for (uint32_t j = 0; j < m; ++j) {
            noc_async_read(get_noc_addr(a_off + r0 + i + j, a), la + j * ROW_BYTES, ROW_BYTES);
            noc_async_read(get_noc_addr(b_off + r0 + i + j, b), lb + j * ROW_BYTES, ROW_BYTES);
        }
        noc_async_read_barrier();
        cb_push_back(tt::CBIndex::c_0, m * TILES);
        cb_push_back(tt::CBIndex::c_1, m * TILES);
    }
}
