// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"

// Copy row 0 of each src tile (two 64 B faces) into in-tile row `row` of the dest gather buffer.
FORCE_INLINE void scatter_row(
    uint32_t src_l1, uint32_t dst_l1, uint32_t dst_x, uint32_t dst_y, uint32_t row, uint32_t Wt, uint32_t page) {
    const uint32_t face_r = (row < 16u) ? 0u : 2u;
    const uint32_t in_face = (row & 15u) * 64u;
    for (uint32_t c = 0; c < Wt; ++c) {
        const uint32_t s = src_l1 + c * page;
        const uint32_t d = dst_l1 + c * page + face_r * 1024u + in_face;
        noc_async_write(s, get_noc_addr(dst_x, dst_y, d), 64u);
        noc_async_write(s + 1024u, get_noc_addr(dst_x, dst_y, d + 1024u), 64u);
    }
}

void kernel_main() {
    constexpr uint32_t cb_out = 16, cb_acc = 17, cb_part = 18, cb_zero_done = 19, cb_go = 20;
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t page = get_compile_time_arg_val(1);
    constexpr uint32_t with_dgamma = get_compile_time_arg_val(2);
    constexpr uint32_t sem_ready1 = get_compile_time_arg_val(3);
    constexpr uint32_t sem_arrive1 = get_compile_time_arg_val(4);
    constexpr uint32_t sem_ready2 = get_compile_time_arg_val(5);
    constexpr uint32_t sem_arrive2 = get_compile_time_arg_val(6);
    constexpr auto out_args = TensorAccessorArgs<7>();
    constexpr auto dg_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t row_count = get_arg_val<uint32_t>(2);
    const uint32_t dg_addr = get_arg_val<uint32_t>(3);
    const uint32_t role = get_arg_val<uint32_t>(4);    // 0 member, 1 row leader, 2 root
    const uint32_t my_col = get_arg_val<uint32_t>(5);  // logical grid x
    const uint32_t my_row = get_arg_val<uint32_t>(6);  // logical grid y
    const uint32_t my_vx = get_arg_val<uint32_t>(7);
    const uint32_t my_vy = get_arg_val<uint32_t>(8);
    const uint32_t leader_vx = get_arg_val<uint32_t>(9);
    const uint32_t leader_vy = get_arg_val<uint32_t>(10);
    const uint32_t root_vx = get_arg_val<uint32_t>(11);
    const uint32_t root_vy = get_arg_val<uint32_t>(12);
    const uint32_t row_cols = get_arg_val<uint32_t>(13);
    const uint32_t n_leaders = get_arg_val<uint32_t>(14);
    constexpr uint32_t TAB = 15;

    const auto out_acc = TensorAccessor(out_args, out_addr, page);
    const auto dg_acc = TensorAccessor(dg_args, dg_addr, page);

    for (uint32_t r = row_start; r < row_start + row_count; ++r) {
        cb_wait_front(cb_out, Wt);
        const uint32_t l1 = get_read_ptr(cb_out);
        const uint32_t base = r * Wt;
        for (uint32_t c = 0; c < Wt; ++c) {
            noc_async_write(l1 + c * page, out_acc.get_noc_addr(base + c), page);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, Wt);
    }

    if constexpr (with_dgamma) {
        Semaphore<> ready1(sem_ready1), arrive1(sem_arrive1), ready2(sem_ready2), arrive2(sem_arrive2);
        Noc noc;
        const uint32_t gather_l1 = get_write_ptr(cb_acc);
        auto open_and_signal = [&](Semaphore<>& ready, uint32_t n, uint32_t tab_off, bool along_x) {
            cb_wait_front(cb_zero_done, 1);
            cb_pop_front(cb_zero_done, 1);
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t v = get_arg_val<uint32_t>(TAB + tab_off + i);
                if (along_x) {
                    ready.up(noc, v, my_vy, 1);
                } else {
                    ready.up(noc, my_vx, v, 1);
                }
            }
        };
        auto go = [&]() {
            cb_reserve_back(cb_go, 1);
            cb_push_back(cb_go, 1);
        };

        cb_wait_front(cb_part, Wt);
        if (role >= 1) {
            open_and_signal(ready1, row_cols, /*tab_off=*/0, /*along_x=*/true);
        }
        ready1.wait_min(1);
        scatter_row(get_read_ptr(cb_part), gather_l1, leader_vx, leader_vy, my_col, Wt, page);
        noc_async_write_barrier();
        arrive1.up(noc, leader_vx, leader_vy, 1);
        cb_pop_front(cb_part, Wt);

        if (role >= 1) {
            arrive1.wait_min(row_cols);
            go();
            cb_wait_front(cb_part, Wt);
            if (role == 2) {
                open_and_signal(ready2, n_leaders, /*tab_off=*/row_cols, /*along_x=*/false);
            }
            ready2.wait_min(1);
            scatter_row(get_read_ptr(cb_part), gather_l1, root_vx, root_vy, my_row, Wt, page);
            noc_async_write_barrier();
            arrive2.up(noc, root_vx, root_vy, 1);
            cb_pop_front(cb_part, Wt);
        }
        if (role == 2) {
            arrive2.wait_min(n_leaders);
            go();
            cb_wait_front(cb_part, Wt);
            const uint32_t l1 = get_read_ptr(cb_part);
            for (uint32_t c = 0; c < Wt; ++c) {
                noc_async_write(l1 + c * page, dg_acc.get_noc_addr(c), page);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_part, Wt);
        }

        // Program-cache hits (and trace replay) reuse this program without re-running host
        // semaphore init. wait_min does not decrement, so leave these at 0 for the next launch.
        ready1.set(0);
        arrive1.set(0);
        ready2.set(0);
        arrive2.set(0);
    }
}
