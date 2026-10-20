// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// mhc_post stats reader (NCRISC, NoC 0). Per unit (mhc_post_stats_common.hpp): when the block changes, the W block
// (n x B tiles, tile k B + c = W row-tile k Ct + b B + c) -> cb_w; then the unit's columns, F[r, c] and
// X_0..X_{n-1}[r, c] of every column c, -> cb_in in one batch of (n + 1) B tiles (one barrier: many reads in flight).
// Root phase: wait until every partial of the rooted rows has landed (the semaphore counts NB per row), then each
// rooted row's NB scratch pages, in block order -> cb_part.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "mhc_post_stats_common.hpp"

void kernel_main() {
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t ct = get_compile_time_arg_val(1);
    constexpr uint32_t rt = get_compile_time_arg_val(2);
    constexpr uint32_t bw = get_compile_time_arg_val(3);
    constexpr uint32_t nb = get_compile_time_arg_val(4);
    constexpr uint32_t page = get_compile_time_arg_val(5);
    constexpr uint32_t cb_in = get_compile_time_arg_val(7);
    constexpr uint32_t cb_w = get_compile_time_arg_val(8);
    constexpr uint32_t cb_part = get_compile_time_arg_val(9);
    constexpr uint32_t sem_id = get_compile_time_arg_val(10);
    constexpr auto f_args = TensorAccessorArgs<11>();
    constexpr auto x_args = TensorAccessorArgs<f_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();

    const auto f_acc = TensorAccessor(f_args, get_arg_val<uint32_t>(0), page);
    const auto x_acc = TensorAccessor(x_args, get_arg_val<uint32_t>(1), page);
    const auto w_acc = TensorAccessor(w_args, get_arg_val<uint32_t>(2), page);
    const auto s_acc = TensorAccessor(s_args, get_arg_val<uint32_t>(3), page);
    const uint32_t start = get_arg_val<uint32_t>(4);
    const uint32_t count = get_arg_val<uint32_t>(5);
    const uint32_t n_rooted = get_arg_val<uint32_t>(6);

    uint32_t cur_b = 0xFFFFFFFFu;
    for (uint32_t u = start; u < start + count; ++u) {
        const mhc_stats::Unit U = mhc_stats::unit_of(u, rt, bw, ct);
        if (U.b != cur_b) {
            cb_reserve_back(cb_w, n * bw);
            const uint32_t base = get_write_ptr(cb_w);
            for (uint32_t k = 0; k < n; ++k) {
                for (uint32_t c = 0; c < U.valid; ++c) {
                    noc_async_read_page(k * ct + U.col0 + c, w_acc, base + (k * bw + c) * page);
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_w, n * bw);
            cur_b = U.b;
        }
        // the unit's B columns in one batch (one barrier): column c's F at slot c (n + 1), X_j at c (n + 1) + 1 + j;
        // always B columns reserved / pushed so the compute's tile indices never wrap the CB
        cb_reserve_back(cb_in, (n + 1) * bw);
        const uint32_t base = get_write_ptr(cb_in);
        for (uint32_t c = 0; c < U.valid; ++c) {
            uint32_t dst = base + c * (n + 1) * page;
            noc_async_read_page(U.r * ct + U.col0 + c, f_acc, dst);
            for (uint32_t j = 0; j < n; ++j) {
                dst += page;
                noc_async_read_page(U.r * n * ct + j * ct + U.col0 + c, x_acc, dst);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, (n + 1) * bw);
    }

    if (n_rooted) {
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
        noc_semaphore_wait_min(sem, n_rooted * nb);
        for (uint32_t i = 0; i < n_rooted; ++i) {
            const uint32_t r = get_arg_val<uint32_t>(7 + i);
            cb_reserve_back(cb_part, nb);  // the row's partials in one batch
            const uint32_t base = get_write_ptr(cb_part);
            for (uint32_t b = 0; b < nb; ++b) {
                noc_async_read_page(r * nb + b, s_acc, base + b * page);
            }
            noc_async_read_barrier();
            cb_push_back(cb_part, nb);
        }
        noc_semaphore_set(sem, 0);
    }
}
