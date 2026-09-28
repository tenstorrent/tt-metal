// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

FORCE_INLINE void zero_tail_rows(const Noc& noc, const CircularBuffer& cb, uint32_t num_tiles, uint32_t valid) {
    constexpr uint32_t face = 1024u, face_row = 64u, tile = 4u * face;
    const uint32_t face_r = (valid < 16u) ? 0u : 2u;
    const uint32_t skip = (valid & 15u) * face_row;
    for (uint32_t t = 0; t < num_tiles; ++t) {
        const uint32_t base = t * tile;
        noc.async_write_zeros(cb, face - skip, {.offset_bytes = base + face_r * face + skip});
        noc.async_write_zeros(cb, (3u - face_r) * face - skip, {.offset_bytes = base + (face_r + 1u) * face + skip});
    }
}

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t page = get_compile_time_arg_val(1);
    constexpr uint32_t with_dgamma = get_compile_time_arg_val(2);
    constexpr auto dy_args = TensorAccessorArgs<3>();
    constexpr auto x_args = TensorAccessorArgs<dy_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto inv_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto d_args = TensorAccessorArgs<inv_args.next_compile_time_args_offset()>();
    static_assert(page == 4096u, "zero_tail_rows assumes 32x32 fp32 tiles");

    const uint32_t dy_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t g_addr = get_arg_val<uint32_t>(2);
    const uint32_t inv_addr = get_arg_val<uint32_t>(3);
    const uint32_t d_addr = get_arg_val<uint32_t>(4);
    const uint32_t row_start = get_arg_val<uint32_t>(5);
    const uint32_t row_count = get_arg_val<uint32_t>(6);
    const uint32_t Ht = get_arg_val<uint32_t>(7);
    const uint32_t h_tail = get_arg_val<uint32_t>(8);

    const auto dy_acc = TensorAccessor(dy_args, dy_addr, page);
    const auto x_acc = TensorAccessor(x_args, x_addr, page);
    const auto g_acc = TensorAccessor(g_args, g_addr, page);
    const auto inv_acc = TensorAccessor(inv_args, inv_addr, page);
    const auto d_acc = TensorAccessor(d_args, d_addr, page);

    const Noc noc;
    CircularBuffer cb_dy(0), cb_x(1), cb_gamma(2), cb_inv(3), cb_d(4);

    if constexpr (with_dgamma) {
        cb_gamma.reserve_back(Wt);
        for (uint32_t c = 0; c < Wt; ++c) {
            noc.async_read(g_acc, cb_gamma, page, {.page_id = c}, {.offset_bytes = c * page});
        }
        noc.async_read_barrier();
        cb_gamma.push_back(Wt);
    }

    for (uint32_t r = row_start; r < row_start + row_count; ++r) {
        cb_inv.reserve_back(1);
        cb_d.reserve_back(1);
        noc.async_read(inv_acc, cb_inv, page, {.page_id = r}, {.offset_bytes = 0});
        noc.async_read(d_acc, cb_d, page, {.page_id = r}, {.offset_bytes = 0});

        cb_dy.reserve_back(Wt);
        cb_x.reserve_back(Wt);
        const uint32_t base = r * Wt;
        for (uint32_t c = 0; c < Wt; ++c) {
            noc.async_read(dy_acc, cb_dy, page, {.page_id = base + c}, {.offset_bytes = c * page});
            noc.async_read(x_acc, cb_x, page, {.page_id = base + c}, {.offset_bytes = c * page});
        }
        noc.async_read_barrier();

        if constexpr (with_dgamma) {
            if (h_tail != 0 && r % Ht == Ht - 1) {
                zero_tail_rows(noc, cb_dy, Wt, h_tail);
                zero_tail_rows(noc, cb_x, Wt, h_tail);
                zero_tail_rows(noc, cb_inv, 1, h_tail);
                noc.write_zeros_l1_barrier();
            }
        }

        cb_inv.push_back(1);
        cb_d.push_back(1);
        cb_dy.push_back(Wt);
        cb_x.push_back(Wt);
    }
}
