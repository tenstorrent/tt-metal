// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// mhc_post stats reader (NCRISC, NoC 0). Per unit (mhc_post_stats_common.hpp): when the block changes, the W block
// (n x B tiles, tile k B + c = W row-tile k Ct + b B + c) -> cb_w; then the unit's columns, F[r, c] and
// X_0..X_{n-1}[r, c] of every column c, -> cb_in in one batch of (n + 1) B tiles (one barrier: many reads in flight).
// Root phase: wait until every partial of the rooted rows has landed (the semaphore counts NB per row), then each
// rooted row's NB scratch pages, in block order -> cb_part.
// MHC_POST_NORM (P.35): per unit also the norm weight's columns (row 0 of each column tile: two 64-byte reads) laid
// out as the unit's weight tile (cb_ng: slot 4 c + g, lane l = w[col0 + c, 2 (l & 7) + (g & 1) + 16 (g >> 1)], the
// columns of a DEST vector v with g = (v & 1) | ((v >> 3) & 1) << 1). A root, after its partials: each rooted row's 32
// rstd values (the compute's cb_nrs, column n (n + 2) + 1) -> 128 bytes at 2 KB into page j of cb_nr on the owner of
// each of the row's NB units (j = the unit's index there), then one semaphore increment each. Every core with units
// then waits for one increment a unit, lays each unit's 32 values out as its rstd tile (slot blk, lane l = rstd of
// token row 4 blk + (l >> 3)) and pushes them.

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

#ifdef MHC_POST_STATS_F16
    // P.19: F bf16 -> its own CB (cb_f, B tiles a unit); the F slots of cb_in stay unused
    constexpr uint32_t cb_f = get_compile_time_arg_val(s_args.next_compile_time_args_offset());
    constexpr uint32_t f_page = get_compile_time_arg_val(s_args.next_compile_time_args_offset() + 1);
    const auto f_acc = TensorAccessor(f_args, get_arg_val<uint32_t>(0), f_page);
#else
    const auto f_acc = TensorAccessor(f_args, get_arg_val<uint32_t>(0), page);
#endif
#ifdef MHC_POST_STATS_X16
    constexpr uint32_t in_page = page / 2;  // P.20: bf16 X tiles in cb_in (the F slots stay unused)
#else
    constexpr uint32_t in_page = page;
#endif
    const auto x_acc = TensorAccessor(x_args, get_arg_val<uint32_t>(1), in_page);
    const auto w_acc = TensorAccessor(w_args, get_arg_val<uint32_t>(2), page);
    const auto s_acc = TensorAccessor(s_args, get_arg_val<uint32_t>(3), page);
    const uint32_t start = get_arg_val<uint32_t>(4);
    const uint32_t count = get_arg_val<uint32_t>(5);
    const uint32_t n_rooted = get_arg_val<uint32_t>(6);
#ifdef MHC_POST_NORM
#ifdef MHC_POST_STATS_F16
    constexpr uint32_t nx0 = s_args.next_compile_time_args_offset() + 2;
#else
    constexpr uint32_t nx0 = s_args.next_compile_time_args_offset();
#endif
    constexpr uint32_t cb_ng = get_compile_time_arg_val(nx0);
    constexpr uint32_t cb_nr = get_compile_time_arg_val(nx0 + 1);
    constexpr uint32_t cb_gstage = get_compile_time_arg_val(nx0 + 2);
    constexpr uint32_t sem_norm_id = get_compile_time_arg_val(nx0 + 3);
    constexpr uint32_t cb_nrs = get_compile_time_arg_val(nx0 + 4);
    constexpr uint32_t cb_nstage = get_compile_time_arg_val(nx0 + 5);
    constexpr auto g_args = TensorAccessorArgs<nx0 + 6>();
    const uint32_t nr_base = get_write_ptr(cb_nr);  // (cb_nr's base, the same on every core: nothing pushed yet)
    const uint32_t sem_norm_addr = get_semaphore(sem_norm_id);
    const uint32_t owners_arg0 = 7 + n_rooted;  // per rooted row, its NB units' owners and slots: j << 24 | x << 16 | y
    const auto g_acc = TensorAccessor(g_args, get_common_arg_val<uint32_t>(0), page);
    using ptr_t = volatile tt_l1_ptr uint32_t*;
    cb_reserve_back(cb_gstage, 1);  // private: never pushed
    const uint32_t gstage = get_write_ptr(cb_gstage);
#endif

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
#ifdef MHC_POST_STATS_F16
        cb_reserve_back(cb_f, bw);
        const uint32_t f_base = get_write_ptr(cb_f);
#endif
        for (uint32_t c = 0; c < U.valid; ++c) {
            uint32_t dst = base + c * (n + 1) * in_page;
#ifdef MHC_POST_STATS_F16
            noc_async_read_page(U.r * ct + U.col0 + c, f_acc, f_base + c * f_page);
#else
            noc_async_read_page(U.r * ct + U.col0 + c, f_acc, dst);
#endif
            for (uint32_t j = 0; j < n; ++j) {
                dst += in_page;
                noc_async_read_page(U.r * n * ct + j * ct + U.col0 + c, x_acc, dst);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, (n + 1) * bw);
#ifdef MHC_POST_STATS_F16
        cb_push_back(cb_f, bw);
#endif
#ifdef MHC_POST_NORM
        {  // the unit's weight tile (row 0 of each column tile: face 0 / face 1 first rows)
            for (uint32_t c = 0; c < U.valid; ++c) {
                noc_async_read(g_acc.get_noc_addr(U.col0 + c, 0), gstage + c * 128, 64);
                noc_async_read(g_acc.get_noc_addr(U.col0 + c, 1024), gstage + c * 128 + 64, 64);
            }
            cb_reserve_back(cb_ng, 1);
            noc_async_read_barrier();
            const ptr_t src = reinterpret_cast<ptr_t>(gstage);
            ptr_t dst = reinterpret_cast<ptr_t>(get_write_ptr(cb_ng));
            for (uint32_t c = 0; c < U.valid; ++c) {
#pragma GCC unroll 1
                for (uint32_t g = 0; g < 4; ++g) {
                    const uint32_t k = 4 * c + g;
                    const uint32_t col = (g & 1u) + 16u * (g >> 1);
#pragma GCC unroll 8
                    for (uint32_t l = 0; l < 32; ++l) {
                        dst[mhc_stats::slot_index(k, l)] = src[c * 32 + col + 2 * (l & 7u)];
                    }
                }
            }
            cb_push_back(cb_ng, 1);
        }
#endif
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
#ifdef MHC_POST_NORM
    if (n_rooted) {  // the roots: each rooted row's 32 rstd values (column n (n + 2) + 1 of the compute's rstd tile) ->
                     // 128 bytes at 2 KB into page j of cb_nr on the owner of each of the row's NB units, then one
                     // increment each (this kernel is idle here; the writer transposes the row for the hc meanwhile)
        uint32_t* stage = reinterpret_cast<uint32_t*>(get_write_ptr(cb_nstage));  // private page (never pushed)
        for (uint32_t i = 0; i < n_rooted; ++i) {
            cb_wait_front(cb_nrs, 1);
            const uint32_t* rs = reinterpret_cast<const uint32_t*>(get_read_ptr(cb_nrs));
            asm volatile("" ::: "memory");
#pragma GCC unroll 8
            for (uint32_t l = 0; l < 32; ++l) {
                stage[l] = rs[mhc_stats::rc_index(l, n * (n + 2) + 1)];
            }
            asm volatile("" ::: "memory");
            cb_pop_front(cb_nrs, 1);
            for (uint32_t b = 0; b < nb; ++b) {
                const uint32_t a = get_arg_val<uint32_t>(owners_arg0 + i * nb + b);
                noc_async_write(
                    reinterpret_cast<uint32_t>(stage),
                    get_noc_addr((a >> 16) & 0xFFu, a & 0xFFFFu, nr_base + (a >> 24) * page + 2048),
                    128);
            }
            noc_async_write_barrier();  // landed (and the stage is free for the next row)
            for (uint32_t b = 0; b < nb; ++b) {
                const uint32_t a = get_arg_val<uint32_t>(owners_arg0 + i * nb + b);
                noc_semaphore_inc(get_noc_addr((a >> 16) & 0xFFu, a & 0xFFFFu, sem_norm_addr), 1);
            }
        }
        noc_async_atomic_barrier();
    }
    if (count) {
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_norm_id));
        cb_reserve_back(cb_nr, count);  // (the CB's base: nothing pushed to it before)
        noc_semaphore_wait_min(sem, count);
        // each unit's 32 rstd values (2 KB into its page) -> its rstd tile: slot blk, lane l = rstd of token row
        // 4 blk + (l >> 3) (the rows of DEST vector v, blk = 4 (v >> 4) + ((v >> 1) & 3)); slots 0..7: the first 1 KB
        const uint32_t base = get_write_ptr(cb_nr);
        asm volatile("" ::: "memory");
        for (uint32_t j = 0; j < count; ++j) {
            const uint32_t* src = reinterpret_cast<const uint32_t*>(base + j * page + 2048);
            uint32_t* dst = reinterpret_cast<uint32_t*>(base + j * page);
#pragma GCC unroll 1
            for (uint32_t blk = 0; blk < 8; ++blk) {
#pragma GCC unroll 4
                for (uint32_t g = 0; g < 4; ++g) {
                    const uint32_t v = src[4 * blk + g];
#pragma GCC unroll 8
                    for (uint32_t e = 0; e < 8; ++e) {
                        dst[mhc_stats::slot_index(blk, 8 * g + e)] = v;
                    }
                }
            }
        }
        asm volatile("" ::: "memory");
        cb_push_back(cb_nr, count);
        noc_semaphore_set(sem, 0);
    }
#endif
}
