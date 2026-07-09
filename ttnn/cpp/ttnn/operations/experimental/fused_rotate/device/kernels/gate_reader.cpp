// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for fused_gate: streams a (Wt tiles/row) and gate (Gt tiles/row) per edge tile-row; in
// backward mode (mode==1) also streams the first Ht tiles of b (= x, for silu'). a and b share the
// [E, nsph*H] row stride (Wt tiles); gate is [E, (nsph-1)*H] (Gt tiles).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gate = get_compile_time_arg_val(1);
    constexpr uint32_t cb_b = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr uint32_t Gt = get_compile_time_arg_val(4);
    constexpr uint32_t Ht = get_compile_time_arg_val(5);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t mode = get_compile_time_arg_val(7);

    constexpr auto a_args = TensorAccessorArgs<8>();
    constexpr auto gate_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<gate_args.next_compile_time_args_offset()>();

    uint32_t arg = 0;
    const uint32_t a_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t gate_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t b_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t start_row = get_arg_val<uint32_t>(arg++);
    const uint32_t num_rows = get_arg_val<uint32_t>(arg++);

    const auto a_gen = TensorAccessor(a_args, a_addr, tile_bytes);
    const auto gate_gen = TensorAccessor(gate_args, gate_addr, tile_bytes);
    const auto b_gen = TensorAccessor(b_args, b_addr, tile_bytes);

    for (uint32_t r = 0; r < num_rows; r++) {
        const uint32_t row = start_row + r;
        const uint32_t abase = row * Wt;
        const uint32_t gbase = row * Gt;

        cb_reserve_back(cb_a, Wt);
        uint32_t aw = get_write_ptr(cb_a);
        for (uint32_t t = 0; t < Wt; t++) {
            noc_async_read_tile(abase + t, a_gen, aw);
            aw += tile_bytes;
        }

        cb_reserve_back(cb_gate, Gt);
        uint32_t gw = get_write_ptr(cb_gate);
        for (uint32_t t = 0; t < Gt; t++) {
            noc_async_read_tile(gbase + t, gate_gen, gw);
            gw += tile_bytes;
        }

        if (mode == 1) {
            cb_reserve_back(cb_b, Ht);
            uint32_t bw = get_write_ptr(cb_b);
            for (uint32_t t = 0; t < Ht; t++) {
                noc_async_read_tile(abase + t, b_gen, bw);
                bw += tile_bytes;
            }
        }

        noc_async_read_barrier();
        cb_push_back(cb_a, Wt);
        cb_push_back(cb_gate, Gt);
        if (mode == 1) {
            cb_push_back(cb_b, Ht);
        }
    }
}
