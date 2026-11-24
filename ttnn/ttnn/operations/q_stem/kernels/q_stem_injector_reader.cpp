// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// q_stem x injector, DRAM half (row 0, RISCV_0 / NoC1): the read side of `x_round_multicast_block`.
// For every stage-1 block in send order and every K-block of it, read the round (m row tiles x s1_kb K tiles) of the
// latent into the next cb_x_round slot. Runs up to x_round_depth rounds ahead of the multicast half (RISCV_1 / NoC0,
// q_stem_injector.cpp), so DRAM latency hides behind the receivers' handshakes.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/ttnn/operations/q_stem/kernels/q_stem_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_x_round = get_compile_time_arg_val(0);
    constexpr uint32_t x_tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t Q_t = get_compile_time_arg_val(2);
    constexpr uint32_t s1_kb = get_compile_time_arg_val(3);
    constexpr uint32_t nkb = get_compile_time_arg_val(4);
    constexpr uint32_t B = get_compile_time_arg_val(5);
    constexpr uint32_t qrt = get_compile_time_arg_val(6);
    constexpr uint32_t m = get_compile_time_arg_val(7);
    constexpr uint32_t G = get_compile_time_arg_val(8);
    constexpr auto x_args = TensorAccessorArgs<10>();
    constexpr uint32_t round_tiles = m * s1_kb;
    constexpr uint32_t St = G * qrt;

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t p = get_arg_val<uint32_t>(1);
    uint32_t rorder[G];
    for (uint32_t i = 0; i + 1 < G; ++i) {
        rorder[i] = get_arg_val<uint32_t>(2 + i);
    }
    const auto x = TensorAccessor(x_args, x_addr, x_tile_bytes);

    q_stem::for_each_block(B, qrt, m, G, p, rorder, [&](uint32_t, uint32_t b, uint32_t d, uint32_t r0) {
        const uint32_t row0 = b * St + d * qrt + r0;
        for (uint32_t kb = 0; kb < nkb; ++kb) {
            cb_reserve_back(cb_x_round, round_tiles);
            uint32_t off = get_write_ptr(cb_x_round);
            for (uint32_t i = 0; i < m; ++i) {
                const uint32_t page0 = (row0 + i) * Q_t + kb * s1_kb;
                for (uint32_t k = 0; k < s1_kb; ++k) {
#ifndef ABLATE_X_READ
                    noc_async_read(x.get_noc_addr(page0 + k), off, x_tile_bytes);
#endif
                    off += x_tile_bytes;
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_x_round, round_tiles);
        }
    });
}
