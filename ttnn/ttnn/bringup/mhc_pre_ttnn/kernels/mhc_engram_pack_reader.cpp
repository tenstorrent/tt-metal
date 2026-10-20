// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// mhc_pre_xing_pack engram reader (NCRISC, NoC 0). Per unit (mhc_engram_pack_common.hpp): when the (stream, block)
// changes, the weight block (B tiles: weight page i Ct + g B + c) -> cb_w; then the unit's h / k tiles in one batch
// (h_i[r, c] at slot 2c, k_i[r, c] at 2c + 1; one barrier) -> cb_in, always 2 B slots so indices never wrap. Root
// phase: wait until the n G partials of every rooted row have landed, then read them in unit order -> cb_part.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "mhc_engram_pack_common.hpp"

void kernel_main() {
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t ct = get_compile_time_arg_val(1);
    constexpr uint32_t kt = get_compile_time_arg_val(2);
    constexpr uint32_t rt = get_compile_time_arg_val(3);
    constexpr uint32_t bw = get_compile_time_arg_val(4);
    constexpr uint32_t groups = get_compile_time_arg_val(5);
    constexpr uint32_t page = get_compile_time_arg_val(6);
    constexpr uint32_t cb_in = get_compile_time_arg_val(7);
    constexpr uint32_t cb_w = get_compile_time_arg_val(8);
    constexpr uint32_t cb_part = get_compile_time_arg_val(9);
    constexpr uint32_t sem_id = get_compile_time_arg_val(10);
    constexpr auto h_args = TensorAccessorArgs<11>();
    constexpr auto k_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr uint32_t nu = n * groups;

    const auto h_acc = TensorAccessor(h_args, get_arg_val<uint32_t>(0), page);
    const auto k_acc = TensorAccessor(k_args, get_arg_val<uint32_t>(1), page);
    const auto w_acc = TensorAccessor(w_args, get_arg_val<uint32_t>(2), page);
    const auto s_acc = TensorAccessor(s_args, get_arg_val<uint32_t>(3), page);
    const uint32_t start = get_arg_val<uint32_t>(4);
    const uint32_t count = get_arg_val<uint32_t>(5);
    const uint32_t n_rooted = get_arg_val<uint32_t>(6);

    uint32_t cur = 0xFFFFFFFFu;
    for (uint32_t u = start; u < start + count; ++u) {
        const mhc_engram::Unit U = mhc_engram::unit_of(u, rt, groups, bw, ct);
        if (U.bu != cur) {
            cb_reserve_back(cb_w, bw);
            const uint32_t base = get_write_ptr(cb_w);
            for (uint32_t c = 0; c < U.valid; ++c) {
                noc_async_read_page(U.i * ct + U.col0 + c, w_acc, base + c * page);
            }
            noc_async_read_barrier();
            cb_push_back(cb_w, bw);
            cur = U.bu;
        }
        cb_reserve_back(cb_in, 2 * bw);
        const uint32_t base = get_write_ptr(cb_in);
        for (uint32_t c = 0; c < U.valid; ++c) {
            noc_async_read_page(U.r * n * ct + U.i * ct + U.col0 + c, h_acc, base + 2 * c * page);
            noc_async_read_page(U.r * kt + U.i * ct + U.col0 + c, k_acc, base + (2 * c + 1) * page);
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, 2 * bw);
    }

    if (n_rooted) {
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
        noc_semaphore_wait_min(sem, n_rooted * nu);
        for (uint32_t j = 0; j < n_rooted; ++j) {
            const uint32_t r = get_arg_val<uint32_t>(7 + j);
            cb_reserve_back(cb_part, nu);  // the row's partials in one batch
            const uint32_t base = get_write_ptr(cb_part);
            for (uint32_t b = 0; b < nu; ++b) {
                noc_async_read_page(r * nu + b, s_acc, base + b * page);
            }
            noc_async_read_barrier();
            cb_push_back(cb_part, nu);
        }
        noc_semaphore_set(sem, 0);
    }
}
