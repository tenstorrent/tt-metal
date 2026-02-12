// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t a_addr = get_arg_val<uint32_t>(0);
    uint32_t b_addr = get_arg_val<uint32_t>(1);
    uint32_t c_addr = get_arg_val<uint32_t>(2);
    uint32_t Mt = get_arg_val<uint32_t>(3);
    uint32_t Kt = get_arg_val<uint32_t>(4);
    uint32_t Nt = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_c = tt::CBIndex::c_2;

    const uint32_t tile_size = get_tile_size(cb_a);

    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, a_addr, tile_size);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr, tile_size);
    constexpr auto c_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    const auto c = TensorAccessor(c_args, c_addr, tile_size);

    for (uint32_t mt = 0; mt < Mt; mt++) {
        for (uint32_t nt = 0; nt < Nt; nt++) {
            // Stream K pairs for matmul
            for (uint32_t kt = 0; kt < Kt; kt++) {
                cb_reserve_back(cb_a, 1);
                uint32_t a_l1 = get_write_ptr(cb_a);
                noc_async_read_tile(mt * Kt + kt, a, a_l1);
                noc_async_read_barrier();
                cb_push_back(cb_a, 1);

                cb_reserve_back(cb_b, 1);
                uint32_t b_l1 = get_write_ptr(cb_b);
                noc_async_read_tile(kt * Nt + nt, b, b_l1);
                noc_async_read_barrier();
                cb_push_back(cb_b, 1);
            }

            // Bias tile
            cb_reserve_back(cb_c, 1);
            uint32_t c_l1 = get_write_ptr(cb_c);
            noc_async_read_tile(mt * Nt + nt, c, c_l1);
            noc_async_read_barrier();
            cb_push_back(cb_c, 1);
        }
    }
}
