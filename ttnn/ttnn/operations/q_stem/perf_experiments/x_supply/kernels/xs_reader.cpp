// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// x_supply bench: injector DRAM/L1 half (the op's q_stem_injector_reader.cpp, tile_aligned path, m = 1).
// Reads this lane's tiles [t_lo, t_lo + n_t) of x row r (= round r) into the next cb_x_round slot (a slot is a whole
// round: round_tiles pages), for the rounds r = r0, r0 + rstep, ...

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t round_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t n_rounds = get_compile_time_arg_val(3);
    constexpr uint32_t r0 = get_compile_time_arg_val(4);
    constexpr uint32_t rstep = get_compile_time_arg_val(5);
    constexpr uint32_t t_lo = get_compile_time_arg_val(6);
    constexpr uint32_t n_t = get_compile_time_arg_val(7);
    constexpr auto x_args = TensorAccessorArgs<8>();
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const auto x = TensorAccessor(x_args, x_addr, tile_bytes);

    for (uint32_t r = r0; r < n_rounds; r += rstep) {
        cb_reserve_back(cb, round_tiles);
        const uint32_t off = get_write_ptr(cb) + t_lo * tile_bytes;
#ifndef XS_NOREAD  // diagnostic: multicast whatever is in the slot (pure multicast rate)
        for (uint32_t t = 0; t < n_t; ++t) {
            noc_async_read(x.get_noc_addr(r * round_tiles + t_lo + t), off + t * tile_bytes, tile_bytes);
        }
#endif
        noc_async_read_barrier();
        cb_push_back(cb, round_tiles);
    }
}
