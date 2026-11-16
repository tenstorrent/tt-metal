// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// moe_decode gate|up writer (RISCV_0, NOC_1): (1) row 0 of the normed activation x ([1,1,1,H] bf16 TILE, 32x32
// tiles) as kt 1x32 tiles for custom_mm (face 0 and face 1 of row 0, 32 bytes each); (2) the slot's routing weight
// to the MATH thread through the mailbox (fp32 bits); (3) the 1x32 h tile to row (slot * pairs + pair) of the h
// buffer ([k * pairs, 32] bf16 row-major, one 64-byte page per tile).
// compile-time args: [cb_x, cb_out, cb_scr, kt, pairs, k, route_page_bytes, (4 unused)]
//                    + TensorAccessorArgs(x, h, route)
// runtime args: [x_addr, h_addr, route_addr, slot, pair, w_addr]
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t h_addr = get_arg_val<uint32_t>(1);
    const uint32_t route_addr = get_arg_val<uint32_t>(2);
    const uint32_t slot = get_arg_val<uint32_t>(3);
    const uint32_t pair = get_arg_val<uint32_t>(4);
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(2);  // this kernel's copy of the routing row
    constexpr uint32_t kt = get_compile_time_arg_val(3);
    constexpr uint32_t pairs = get_compile_time_arg_val(4);
    constexpr uint32_t k = get_compile_time_arg_val(5);
    constexpr uint32_t route_page_bytes = get_compile_time_arg_val(6);
    constexpr auto x_args = TensorAccessorArgs<11>();
    constexpr auto h_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto r_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 2048, face_bytes = 512, row_bytes = 32, tiny_bytes = 64;
    const auto s_x = TensorAccessor(x_args, x_addr, tile_bytes);
    const auto s_h = TensorAccessor(h_args, h_addr, tiny_bytes);
    const auto s_r = TensorAccessor(r_args, route_addr, route_page_bytes);

    // routing row: the slot's weight goes to the MATH thread
    const uint32_t scratch = get_write_ptr(cb_scr);
    noc_async_read(s_r.get_noc_addr(0), scratch, route_page_bytes);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* route = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch);
    cb_reserve_back(cb_x, kt);
    uint32_t l1 = get_write_ptr(cb_x);
    for (uint32_t t = 0; t < kt; ++t) {
        const uint64_t src = s_x.get_noc_addr(t);
        noc_async_read(src, l1, row_bytes);
        noc_async_read(src + face_bytes, l1 + row_bytes, row_bytes);
        l1 += tiny_bytes;
    }
    noc_async_read_barrier();
    const uint16_t w = route[slot];
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, static_cast<uint32_t>(w) << 16);
    cb_push_back(cb_x, kt);

    cb_wait_front(cb_out, 1);
    noc_async_write(get_read_ptr(cb_out), s_h.get_noc_addr(slot * pairs + pair), tiny_bytes);
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
}
