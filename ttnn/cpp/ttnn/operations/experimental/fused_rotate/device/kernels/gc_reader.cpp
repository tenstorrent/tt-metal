// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for the coefficient-adjoint (gc) kernel: streams gout + xin tile-rows and loads the 32
// pos-independent column-selector tiles once (resident for the whole kernel).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_gout = get_compile_time_arg_val(0);
    constexpr uint32_t cb_xin = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sel = get_compile_time_arg_val(2);
    constexpr uint32_t n_out_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t n_in_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(5);

    constexpr auto gout_args = TensorAccessorArgs<6>();
    constexpr auto xin_args = TensorAccessorArgs<gout_args.next_compile_time_args_offset()>();
    constexpr auto sel_args = TensorAccessorArgs<xin_args.next_compile_time_args_offset()>();

    uint32_t arg = 0;
    const uint32_t gout_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t xin_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t sel_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t start_row = get_arg_val<uint32_t>(arg++);
    const uint32_t num_rows = get_arg_val<uint32_t>(arg++);

    const auto gout_gen = TensorAccessor(gout_args, gout_addr, tile_bytes);
    const auto xin_gen = TensorAccessor(xin_args, xin_addr, tile_bytes);
    const auto sel_gen = TensorAccessor(sel_args, sel_addr, tile_bytes);

    // selector tiles 0..31 (one tile-row of the [32, 32*32] constant), loaded once.
    cb_reserve_back(cb_sel, 32);
    uint32_t sw = get_write_ptr(cb_sel);
    for (uint32_t t = 0; t < 32; t++) {
        noc_async_read_tile(t, sel_gen, sw);
        sw += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_sel, 32);

    for (uint32_t r = 0; r < num_rows; r++) {
        const uint32_t row = start_row + r;

        cb_reserve_back(cb_gout, n_out_tiles);
        uint32_t gw = get_write_ptr(cb_gout);
        const uint32_t gout_base = row * n_out_tiles;
        for (uint32_t t = 0; t < n_out_tiles; t++) {
            noc_async_read_tile(gout_base + t, gout_gen, gw);
            gw += tile_bytes;
        }

        cb_reserve_back(cb_xin, n_in_tiles);
        uint32_t xw = get_write_ptr(cb_xin);
        const uint32_t xin_base = row * n_in_tiles;
        for (uint32_t t = 0; t < n_in_tiles; t++) {
            noc_async_read_tile(xin_base + t, xin_gen, xw);
            xw += tile_bytes;
        }

        noc_async_read_barrier();
        cb_push_back(cb_gout, n_out_tiles);
        cb_push_back(cb_xin, n_in_tiles);
    }
}
