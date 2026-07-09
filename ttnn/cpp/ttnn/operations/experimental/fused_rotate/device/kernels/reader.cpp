// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_coef = get_compile_time_arg_val(1);
    constexpr uint32_t n_in_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t coef_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);

    constexpr auto x_args = TensorAccessorArgs<5>();
    const auto coef_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

    uint32_t arg = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t coef_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t start_row = get_arg_val<uint32_t>(arg++);
    const uint32_t num_rows = get_arg_val<uint32_t>(arg++);

    const auto x_gen = TensorAccessor(x_args, x_addr, tile_bytes);
    const auto coef_gen = TensorAccessor(coef_args, coef_addr, tile_bytes);

    for (uint32_t r = 0; r < num_rows; r++) {
        const uint32_t row = start_row + r;

        // input feature blocks: tiles [row*n_in_tiles .. +n_in_tiles)
        cb_reserve_back(cb_x, n_in_tiles);
        uint32_t xw = get_write_ptr(cb_x);
        const uint32_t x_base = row * n_in_tiles;
        for (uint32_t t = 0; t < n_in_tiles; t++) {
            noc_async_read_tile(x_base + t, x_gen, xw);
            xw += tile_bytes;
        }

        // per-nonzero coefficient tiles: tiles [row*coef_tiles .. +coef_tiles)
        cb_reserve_back(cb_coef, coef_tiles);
        uint32_t cw = get_write_ptr(cb_coef);
        const uint32_t coef_base = row * coef_tiles;
        for (uint32_t t = 0; t < coef_tiles; t++) {
            noc_async_read_tile(coef_base + t, coef_gen, cw);
            cw += tile_bytes;
        }

        noc_async_read_barrier();
        cb_push_back(cb_x, n_in_tiles);
        cb_push_back(cb_coef, coef_tiles);
    }
}
