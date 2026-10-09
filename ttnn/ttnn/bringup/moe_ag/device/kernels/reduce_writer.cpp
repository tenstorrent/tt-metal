// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// All-gather MoE local reduce, writer (BRISC): each reduced token row (c_16, TILES tiles) to the partial sums.
// SPLIT (two mesh rows): token g of this chip's row goes to own[g % S], the other row's to other[g % S] (the row
// comes from the per-device chip-info tensor, word 0); else out[g].
// CT: 0 ROW_BYTES, 1 TILES, 2 S (chunk_size_per_chip), 3 SPLIT
// Common RT: 0 own / out addr, 1 other addr, 2 chip-info addr, 3 rows, 4 rows per core (range g0, n), 5 grid x
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "core_range.hpp"

void kernel_main() {
    constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(0);
    constexpr uint32_t TILES = get_compile_time_arg_val(1);
    constexpr uint32_t S = get_compile_time_arg_val(2);
    constexpr bool SPLIT = get_compile_time_arg_val(3) != 0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const auto [g0, n] =
        core_range(get_common_arg_val<uint32_t>(3), get_common_arg_val<uint32_t>(4), get_common_arg_val<uint32_t>(5));
    const InterleavedAddrGen<true> own = {.bank_base_address = get_common_arg_val<uint32_t>(0), .page_size = ROW_BYTES};
    const InterleavedAddrGen<true> other = {
        .bank_base_address = get_common_arg_val<uint32_t>(1), .page_size = ROW_BYTES};
    uint32_t my_row = 0;
    if constexpr (SPLIT) {
        const uint32_t l1 = get_write_ptr(tt::CBIndex::c_7);
        noc_async_read(
            get_noc_addr(
                0, InterleavedAddrGen<true>{.bank_base_address = get_common_arg_val<uint32_t>(2), .page_size = 64}),
            l1,
            64);
        noc_async_read_barrier();
        my_row = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    }
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t g = g0 + i;
        cb_wait_front(cb_out, TILES);
        uint64_t dst;
        if constexpr (SPLIT) {
            dst = g / S == my_row ? get_noc_addr(g % S, own) : get_noc_addr(g % S, other);
        } else {
            dst = get_noc_addr(g, own);
        }
        noc_async_write(get_read_ptr(cb_out), dst, ROW_BYTES);
        noc_async_write_barrier();
        cb_pop_front(cb_out, TILES);
    }
    noc_async_full_barrier();
}
