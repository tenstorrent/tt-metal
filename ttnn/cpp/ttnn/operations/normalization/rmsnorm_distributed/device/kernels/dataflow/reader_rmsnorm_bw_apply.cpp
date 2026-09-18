// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_dy = 0, cb_x = 1, cb_gamma = 2, cb_inv = 3, cb_d = 4;
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t page = get_compile_time_arg_val(1);
    constexpr uint32_t with_dgamma = get_compile_time_arg_val(2);
    constexpr auto dy_args = TensorAccessorArgs<3>();
    constexpr auto x_args = TensorAccessorArgs<dy_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto inv_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto d_args = TensorAccessorArgs<inv_args.next_compile_time_args_offset()>();

    const uint32_t dy_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t g_addr = get_arg_val<uint32_t>(2);
    const uint32_t inv_addr = get_arg_val<uint32_t>(3);
    const uint32_t d_addr = get_arg_val<uint32_t>(4);
    const uint32_t row_start = get_arg_val<uint32_t>(5);
    const uint32_t row_count = get_arg_val<uint32_t>(6);

    const auto dy_acc = TensorAccessor(dy_args, dy_addr, page);
    const auto x_acc = TensorAccessor(x_args, x_addr, page);
    const auto g_acc = TensorAccessor(g_args, g_addr, page);
    const auto inv_acc = TensorAccessor(inv_args, inv_addr, page);
    const auto d_acc = TensorAccessor(d_args, d_addr, page);

    if constexpr (with_dgamma) {
        cb_reserve_back(cb_gamma, Wt);
        {
            const uint32_t l1 = get_write_ptr(cb_gamma);
            for (uint32_t c = 0; c < Wt; ++c) {
                noc_async_read(g_acc.get_noc_addr(c), l1 + c * page, page);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_gamma, Wt);
    }

    for (uint32_t r = row_start; r < row_start + row_count; ++r) {
        cb_reserve_back(cb_inv, 1);
        cb_reserve_back(cb_d, 1);
        noc_async_read(inv_acc.get_noc_addr(r), get_write_ptr(cb_inv), page);
        noc_async_read(d_acc.get_noc_addr(r), get_write_ptr(cb_d), page);

        cb_reserve_back(cb_dy, Wt);
        cb_reserve_back(cb_x, Wt);
        const uint32_t dy_l1 = get_write_ptr(cb_dy);
        const uint32_t x_l1 = get_write_ptr(cb_x);
        const uint32_t base = r * Wt;
        for (uint32_t c = 0; c < Wt; ++c) {
            noc_async_read(dy_acc.get_noc_addr(base + c), dy_l1 + c * page, page);
            noc_async_read(x_acc.get_noc_addr(base + c), x_l1 + c * page, page);
        }
        noc_async_read_barrier();
        cb_push_back(cb_inv, 1);
        cb_push_back(cb_d, 1);
        cb_push_back(cb_dy, Wt);
        cb_push_back(cb_x, Wt);
    }
}
