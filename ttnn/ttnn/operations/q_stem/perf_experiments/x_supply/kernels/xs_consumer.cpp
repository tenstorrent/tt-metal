// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// x_supply bench, stand-in for the stage-1 compute (shared by every variant): per round wait the round in
// cb_x_round, checksum it (FULL: every word; else 4 words per tile, spread over the tile), hold the slot for
// `spin` wall-clock ticks (the matmul's time on the round), pop it. Writes the checksum to its output page.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t round_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t n_rounds = get_compile_time_arg_val(4);
    constexpr uint32_t spin = get_compile_time_arg_val(5);
    constexpr bool full = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t out_page = get_compile_time_arg_val(7);
    constexpr auto out_args = TensorAccessorArgs<8>();
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t my_page = get_arg_val<uint32_t>(1);
    constexpr uint32_t words = tile_bytes / 4;

    uint32_t sum = 0;
    for (uint32_t r = 0; r < n_rounds; ++r) {
        cb_wait_front(cb_x, round_tiles);
        const uint64_t t0 = get_timestamp();
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_x));
        for (uint32_t t = 0; t < round_tiles; ++t) {
            volatile tt_l1_ptr uint32_t* w = p + t * words;
            if constexpr (full) {
                for (uint32_t i = 0; i < words; ++i) {
                    sum += w[i];
                }
            } else {
                sum += w[0] + w[words / 3] + w[2 * words / 3] + w[words - 1];
            }
        }
        if constexpr (spin > 0) {
            while (get_timestamp() - t0 < spin) {
            }
        }
        cb_pop_front(cb_x, round_tiles);
    }
    cb_reserve_back(cb_out, 1);
    const uint32_t o = get_write_ptr(cb_out);
    volatile tt_l1_ptr uint32_t* ow = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(o);
    ow[0] = sum;
    for (uint32_t i = 1; i < out_page / 4; ++i) {
        ow[i] = 0;
    }
    const auto out = TensorAccessor(out_args, out_addr, out_page);
    noc_async_write(o, out.get_noc_addr(my_page), out_page);
    noc_async_write_barrier();
}
