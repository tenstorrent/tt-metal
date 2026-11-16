// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// moe_decode batch-1 fused program, writer (RISCV_0, NOC_1).
//   gate|up role: row 0 of x as kt 1x32 tiles -> cb_x; the slot's routing weight -> MATH thread (mailbox); then the
//     h tile stays in cb_hout (the same L1 address on every core) and each down core's semaphore is incremented;
//     the down cores pull the tiles.
//   down role: the column's 1x32 result into row 0 of output tile `col` (faces 0 and 1 of row 0).
// compile-time args: [cb_x, cb_hout, cb_h, cb_out, cb_scr, kt, pairs, k, route_page_bytes, num_down, h_sem_id,
//                     grid_cores] + TensorAccessorArgs(x, out, route)
// runtime args: [x_addr, out_addr, route_addr, is_gu, slot, pair, has_down, col,
//                (4 unused), down core (x, y) NOC coordinates * num_down]
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t route_addr = get_arg_val<uint32_t>(2);
    const uint32_t is_gu = get_arg_val<uint32_t>(3);
    const uint32_t slot = get_arg_val<uint32_t>(4);
    const uint32_t pair = get_arg_val<uint32_t>(5);
    const uint32_t has_down = get_arg_val<uint32_t>(6);
    const uint32_t col = get_arg_val<uint32_t>(7);
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_hout = get_compile_time_arg_val(1);
    constexpr uint32_t cb_h = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(4);
    constexpr uint32_t kt = get_compile_time_arg_val(5);
    constexpr uint32_t pairs = get_compile_time_arg_val(6);
    constexpr uint32_t k = get_compile_time_arg_val(7);
    constexpr uint32_t route_page_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t num_down = get_compile_time_arg_val(9);
    constexpr uint32_t h_sem_id = get_compile_time_arg_val(10);
    constexpr uint32_t grid_cores = get_compile_time_arg_val(11);
    constexpr auto x_args = TensorAccessorArgs<12>();
    constexpr auto o_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto r_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 2048, face_bytes = 512, row_bytes = 32, tiny_bytes = 64;
    if (is_gu) {
        const auto s_x = TensorAccessor(x_args, x_addr, tile_bytes);
        const auto s_r = TensorAccessor(r_args, route_addr, route_page_bytes);
        const uint32_t scratch = get_write_ptr(cb_scr);
        noc_async_read(s_r.get_noc_addr(0), scratch, route_page_bytes);
        cb_reserve_back(cb_x, kt);
        uint32_t l1 = get_write_ptr(cb_x);
        for (uint32_t t = 0; t < kt; ++t) {
            const uint64_t src = s_x.get_noc_addr(t);
            noc_async_read(src, l1, row_bytes);
            noc_async_read(src + face_bytes, l1 + row_bytes, row_bytes);
            l1 += tiny_bytes;
        }
        noc_async_read_barrier();
        const uint16_t w = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch)[slot];
        ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, static_cast<uint32_t>(w) << 16);
        cb_push_back(cb_x, kt);

        // h stays in cb_hout (same L1 address on every core) for the down cores to pull; signal each down core
        cb_wait_front(cb_hout, 1);
#ifndef FUSED_NO_SYNC
        const uint32_t sem = get_semaphore(h_sem_id);
        for (uint32_t d = 0; d < num_down; ++d) {
            noc_semaphore_inc(
                get_noc_addr(get_arg_val<uint32_t>(12 + 2 * d), get_arg_val<uint32_t>(13 + 2 * d), sem), 1);
        }
        noc_async_atomic_barrier();
#endif
    }
    if (has_down) {
        const auto s_o = TensorAccessor(o_args, out_addr, tile_bytes);
        cb_wait_front(cb_out, 1);
        const uint32_t src = get_read_ptr(cb_out);
        const uint64_t dst = s_o.get_noc_addr(col);
        noc_async_write(src, dst, row_bytes);
        noc_async_write(src + row_bytes, dst + face_bytes, row_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
