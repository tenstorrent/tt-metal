// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for fused_ln_bw (silu+gamma folded): loads the [32,32] reduction tile (col0 = 1/W) and the
// [1,W] LN affine scale (gamma) once (both resident), then streams g_out, x and n tile-rows (Wt tiles
// each) per edge tile-row. The compute kernel builds gy = g_out*silu'(n)*gamma internally.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_gout = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = get_compile_time_arg_val(1);
    constexpr uint32_t cb_red = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t cb_n = get_compile_time_arg_val(5);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(6);

    constexpr auto gout_args = TensorAccessorArgs<7>();
    constexpr auto x_args = TensorAccessorArgs<gout_args.next_compile_time_args_offset()>();
    constexpr auto red_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto n_args = TensorAccessorArgs<red_args.next_compile_time_args_offset()>();
    constexpr auto gamma_args = TensorAccessorArgs<n_args.next_compile_time_args_offset()>();

    uint32_t arg = 0;
    const uint32_t gout_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t x_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t red_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t start_row = get_arg_val<uint32_t>(arg++);
    const uint32_t num_rows = get_arg_val<uint32_t>(arg++);
    const uint32_t n_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(arg++);

    const auto gout_gen = TensorAccessor(gout_args, gout_addr, tile_bytes);
    const auto x_gen = TensorAccessor(x_args, x_addr, tile_bytes);
    const auto red_gen = TensorAccessor(red_args, red_addr, tile_bytes);
    const auto n_gen = TensorAccessor(n_args, n_addr, tile_bytes);
    const auto gamma_gen = TensorAccessor(gamma_args, gamma_addr, tile_bytes);

    // reduction tile 0, loaded once (resident)
    cb_reserve_back(cb_red, 1);
    uint32_t rw = get_write_ptr(cb_red);
    noc_async_read_tile(0, red_gen, rw);
    // gamma row (Wt tiles, tile-row 0), loaded once (resident)
    cb_reserve_back(cb_gamma, Wt);
    uint32_t gaw = get_write_ptr(cb_gamma);
    for (uint32_t t = 0; t < Wt; t++) {
        noc_async_read_tile(t, gamma_gen, gaw);
        gaw += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_red, 1);
    cb_push_back(cb_gamma, Wt);

    for (uint32_t r = 0; r < num_rows; r++) {
        const uint32_t row = start_row + r;
        const uint32_t base = row * Wt;

        cb_reserve_back(cb_gout, Wt);
        uint32_t gw = get_write_ptr(cb_gout);
        for (uint32_t t = 0; t < Wt; t++) {
            noc_async_read_tile(base + t, gout_gen, gw);
            gw += tile_bytes;
        }

        cb_reserve_back(cb_x, Wt);
        uint32_t xw = get_write_ptr(cb_x);
        for (uint32_t t = 0; t < Wt; t++) {
            noc_async_read_tile(base + t, x_gen, xw);
            xw += tile_bytes;
        }

        cb_reserve_back(cb_n, Wt);
        uint32_t nw = get_write_ptr(cb_n);
        for (uint32_t t = 0; t < Wt; t++) {
            noc_async_read_tile(base + t, n_gen, nw);
            nw += tile_bytes;
        }

        noc_async_read_barrier();
        cb_push_back(cb_gout, Wt);
        cb_push_back(cb_x, Wt);
        cb_push_back(cb_n, Wt);
    }
}
