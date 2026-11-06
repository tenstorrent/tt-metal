// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// mhc_post stats writer (BRISC, NoC 1). Per unit (mhc_post_stats_common.hpp):
//   1. the unit's coefficient tiles: post / comb row tile r -> cb_raw, expanded into the two half sets of cb_cf
//      (fp32 word copies, bit-exact; the next unit's set is built before this unit's outputs are drained);
//   2. the unit's X' tiles (cb_out, one batch of n B) -> X' pages r n Ct + k Ct + b B + c;
//   3. the unit's partial row: the lane-wise sum-of-squares tile (cb_sfin) -> coefficient-major (cb_soa); the
//      compute's row totals (cb_soa_out slot 0) -> column ss_col of the X' @ W tile (cb_mfin) -> scratch page r NB + b,
//      then one semaphore increment on row r's root core.
// Root phase: each rooted row's sum (cb_fin) -> stats page r.
// MHC_POST_COLLAPSE (P.34): step 1 also reads the collapse_pre row tile r and builds the unit's pre-block tile (cb_pb,
//   mhc_pre_xing_writer.cpp build_pre_blocks); after the X' tiles, the unit's y tiles (cb_y) -> y pages r Ct + b B + c.
//   Root phase: the row's sum (cb_fin) also -> coefficient-major (cb_hsoa); the compute's hc (cb_hsoa_out) -> the
//   row-major stage (columns n (n + 2)..31 zero) -> hc page r. All moves are fp32 word copies (bit-exact).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "mhc_post_stats_common.hpp"

namespace {

using mhc_stats::rc_index;
using mhc_stats::slot_index;
using ptr_t = volatile tt_l1_ptr uint32_t*;

// dst coefficient-major slots [0, 32) <- src row-major columns [0, 32) (slot k = column k, lane l = row l)
inline void cols_to_soa(const volatile tt_l1_ptr uint32_t* src, volatile tt_l1_ptr uint32_t* dst) {
#pragma GCC unroll 1
    for (uint32_t l = 0; l < 32; ++l) {
        const volatile tt_l1_ptr uint32_t* s = src + rc_index(l, 0);
        volatile tt_l1_ptr uint32_t* d = dst + slot_index(0, l);
#pragma GCC unroll 32
        for (uint32_t k = 0; k < 32; ++k) {
            d[slot_index(k, 0)] = s[rc_index(0, k)];
        }
    }
}

// The coefficient tiles of one unit (both halves, 2 * CF tiles at dst) from the raw post / comb row tiles.
template <uint32_t N, uint32_t CF, bool COMB_T>
inline void build_coefficients(
    const volatile tt_l1_ptr uint32_t* post,
    const volatile tt_l1_ptr uint32_t* comb,
    volatile tt_l1_ptr uint32_t* dst) {
    constexpr uint32_t TERMS = N * (N + 1);
#pragma GCC unroll 1
    for (uint32_t h = 0; h < 2; ++h) {
        volatile tt_l1_ptr uint32_t* half = dst + h * CF * 1024u;
#pragma GCC unroll 1
        for (uint32_t t = 0; t < TERMS; ++t) {
            const uint32_t k = t / (N + 1), m = t % (N + 1);
            const volatile tt_l1_ptr uint32_t* src = m == 0 ? post : comb;
#ifdef MHC_POST_POST_COL
            // P.19: post / comb at column offsets of a wider tile row (the packed mixes)
            const uint32_t col =
                m == 0 ? MHC_POST_POST_COL + k : MHC_POST_COMB_COL + (COMB_T ? (m - 1) * N + k : k * N + (m - 1));
#else
            const uint32_t col = m == 0 ? k : (COMB_T ? (m - 1) * N + k : k * N + (m - 1));
#endif
#pragma GCC unroll 1
            for (uint32_t q4 = 0; q4 < 4; ++q4) {
                const uint32_t q = 4 * t + q4;
                volatile tt_l1_ptr uint32_t* d = half + (q >> 5) * 1024u + slot_index(q & 31u, 0);
                const uint32_t row0 = 4 * (4 * h + q4);
#pragma GCC unroll 4
                for (uint32_t g = 0; g < 4; ++g) {
                    const uint32_t v = src[rc_index(row0 + g, col)];
#pragma GCC unroll 8
                    for (uint32_t e = 0; e < 8; ++e) {
                        d[slot_index(0, 8 * g + e)] = v;
                    }
                }
            }
        }
    }
}

#ifdef MHC_POST_COLLAPSE
// Pre-block tile (mhc_pre_xing_writer.cpp, the same words): slot 8 i + b, lane l <- pre_i (row-major column i) of
// token row 4 b + (l >> 3).
template <uint32_t N>
inline void build_pre_blocks(const volatile tt_l1_ptr uint32_t* src, volatile tt_l1_ptr uint32_t* dst) {
#pragma GCC unroll 1
    for (uint32_t q = 0; q < 8 * N; ++q) {
        const uint32_t i = q >> 3, b = q & 7u;
        volatile tt_l1_ptr uint32_t* d = dst + slot_index(q, 0);
#pragma GCC unroll 1
        for (uint32_t g = 0; g < 4; ++g) {  // lanes 8 g .. 8 g + 7 share token row 4 b + g
            const uint32_t v = src[rc_index(4 * b + g, i)];
#pragma GCC unroll 8
            for (uint32_t e = 0; e < 8; ++e) {
                d[2u * (8u * g + e)] = v;
            }
        }
    }
}

// dst row-major columns [0, COUNT) <- src coefficient-major slots [0, COUNT)
template <uint32_t COUNT>
inline void soa_to_cols(const volatile tt_l1_ptr uint32_t* src, volatile tt_l1_ptr uint32_t* dst) {
#pragma GCC unroll 1
    for (uint32_t l = 0; l < 32; ++l) {
        const volatile tt_l1_ptr uint32_t* s = src + slot_index(0, l);
        volatile tt_l1_ptr uint32_t* d = dst + rc_index(l, 0);
#pragma GCC unroll 32
        for (uint32_t k = 0; k < COUNT; ++k) {
            d[rc_index(0, k)] = s[slot_index(k, 0)];
        }
    }
}
#endif

}  // namespace

void kernel_main() {
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t ct = get_compile_time_arg_val(1);
    constexpr uint32_t rt = get_compile_time_arg_val(2);
    constexpr uint32_t bw = get_compile_time_arg_val(3);
    constexpr uint32_t nb = get_compile_time_arg_val(4);
    constexpr uint32_t page = get_compile_time_arg_val(5);
    constexpr uint32_t cf = get_compile_time_arg_val(6);
    constexpr uint32_t cb_out = get_compile_time_arg_val(7);
    constexpr uint32_t cb_cf = get_compile_time_arg_val(8);
    constexpr uint32_t cb_raw = get_compile_time_arg_val(9);
    constexpr uint32_t cb_mfin = get_compile_time_arg_val(10);
    constexpr uint32_t cb_sfin = get_compile_time_arg_val(11);
    constexpr uint32_t cb_soa = get_compile_time_arg_val(12);
    constexpr uint32_t cb_soa_out = get_compile_time_arg_val(13);
    constexpr uint32_t cb_fin = get_compile_time_arg_val(14);
    constexpr uint32_t sem_id = get_compile_time_arg_val(15);
    constexpr uint32_t ss_col = get_compile_time_arg_val(16);
    constexpr bool comb_t = get_compile_time_arg_val(17) != 0;
    constexpr auto o_args = TensorAccessorArgs<18>();
    constexpr auto p_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto st_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();

#ifdef MHC_POST_STATS_X16
    constexpr uint32_t o_page = page / 2;  // P.20: bf16 X' tiles
#else
    constexpr uint32_t o_page = page;
#endif
    const auto o_acc = TensorAccessor(o_args, get_arg_val<uint32_t>(0), o_page);
    const auto p_acc = TensorAccessor(p_args, get_arg_val<uint32_t>(1), page);
    const auto c_acc = TensorAccessor(c_args, get_arg_val<uint32_t>(2), page);
    const auto s_acc = TensorAccessor(s_args, get_arg_val<uint32_t>(3), page);
    const auto st_acc = TensorAccessor(st_args, get_arg_val<uint32_t>(4), page);
    const uint32_t start = get_arg_val<uint32_t>(5);
    const uint32_t count = get_arg_val<uint32_t>(6);
    const uint32_t n_rooted = get_arg_val<uint32_t>(7);
    const uint32_t root_arg0 = 8 + n_rooted;  // root core (x << 16 | y) of each unit's row
    const uint32_t sem_addr = get_semaphore(sem_id);
#ifdef MHC_POST_COLLAPSE
    constexpr uint32_t cx0 = st_args.next_compile_time_args_offset();
    constexpr uint32_t cb_pb = get_compile_time_arg_val(cx0);
    constexpr uint32_t cb_y = get_compile_time_arg_val(cx0 + 1);
    constexpr uint32_t cb_hsoa = get_compile_time_arg_val(cx0 + 2);
    constexpr uint32_t cb_hsoa_out = get_compile_time_arg_val(cx0 + 3);
    constexpr uint32_t cb_hstage = get_compile_time_arg_val(cx0 + 4);
    constexpr auto pre_args = TensorAccessorArgs<cx0 + 5>();
    constexpr auto hc_args = TensorAccessorArgs<pre_args.next_compile_time_args_offset()>();
    constexpr auto y_args = TensorAccessorArgs<hc_args.next_compile_time_args_offset()>();
    constexpr uint32_t hc_cols = n * (n + 2);
    // the collapse_pre, hc and y addresses: common runtime args (the same on every core; DS41-145, DS41-210)
    const auto pre_acc = TensorAccessor(pre_args, get_common_arg_val<uint32_t>(0), page);
    const auto hc_acc = TensorAccessor(hc_args, get_common_arg_val<uint32_t>(1), page);
    const auto y_acc = TensorAccessor(y_args, get_common_arg_val<uint32_t>(2), o_page);
#endif

    auto build = [&](uint32_t u) {
        const mhc_stats::Unit U = mhc_stats::unit_of(u, rt, bw, ct);
        cb_reserve_back(cb_raw, 2);
        const uint32_t raw = get_write_ptr(cb_raw);
        noc_async_read_page(U.r, p_acc, raw);
        noc_async_read_page(U.r, c_acc, raw + page);
#ifdef MHC_POST_COLLAPSE
        noc_async_read_page(U.r, pre_acc, raw + 2 * page);
#endif
        noc_async_read_barrier();
        cb_reserve_back(cb_cf, 2 * cf);
        build_coefficients<n, cf, comb_t>(
            reinterpret_cast<ptr_t>(raw),
            reinterpret_cast<ptr_t>(raw + page),
            reinterpret_cast<ptr_t>(get_write_ptr(cb_cf)));
        cb_push_back(cb_cf, 2 * cf);
#ifdef MHC_POST_COLLAPSE
        cb_reserve_back(cb_pb, 1);
        build_pre_blocks<n>(reinterpret_cast<ptr_t>(raw + 2 * page), reinterpret_cast<ptr_t>(get_write_ptr(cb_pb)));
        cb_push_back(cb_pb, 1);
#endif
    };

    if (count) {
        build(start);
    }
    for (uint32_t u = start; u < start + count; ++u) {
        const mhc_stats::Unit U = mhc_stats::unit_of(u, rt, bw, ct);
        if (u + 1 < start + count) {
            build(u + 1);
        }
        cb_wait_front(cb_out, n * bw);  // the unit's X' tiles, tile c n + k = X'_k[r, b B + c]
        {
            const uint32_t src = get_read_ptr(cb_out);
            for (uint32_t c = 0; c < U.valid; ++c) {
                for (uint32_t k = 0; k < n; ++k) {
                    noc_async_write_page(U.r * n * ct + k * ct + U.col0 + c, o_acc, src + (c * n + k) * o_page);
                }
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_out, n * bw);
        }
#ifdef MHC_POST_COLLAPSE
        cb_wait_front(cb_y, bw);  // the unit's y tiles, tile c = y[r, b B + c]
        {
            const uint32_t src = get_read_ptr(cb_y);
            for (uint32_t c = 0; c < U.valid; ++c) {
                noc_async_write_page(U.r * ct + U.col0 + c, y_acc, src + c * o_page);
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_y, bw);
        }
#endif
        // the unit's partial row
        cb_wait_front(cb_sfin, 1);
        cb_reserve_back(cb_soa, 1);
        cols_to_soa(reinterpret_cast<ptr_t>(get_read_ptr(cb_sfin)), reinterpret_cast<ptr_t>(get_write_ptr(cb_soa)));
        cb_push_back(cb_soa, 1);
        cb_pop_front(cb_sfin, 1);
        cb_wait_front(cb_soa_out, 1);
        cb_wait_front(cb_mfin, 1);
        ptr_t tot = reinterpret_cast<ptr_t>(get_read_ptr(cb_soa_out));
        ptr_t mix = reinterpret_cast<ptr_t>(get_read_ptr(cb_mfin));
        for (uint32_t l = 0; l < 32; ++l) {
            mix[rc_index(l, ss_col)] = tot[slot_index(0, l)];
        }
        cb_pop_front(cb_soa_out, 1);
        noc_async_write_page(U.r * nb + U.b, s_acc, get_read_ptr(cb_mfin));
        noc_async_write_barrier();
        cb_pop_front(cb_mfin, 1);
        const uint32_t xy = get_arg_val<uint32_t>(root_arg0 + (u - start));
        noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFFu, sem_addr), 1);
    }

#ifdef MHC_POST_COLLAPSE
    ptr_t stage = nullptr;
    if (n_rooted) {  // private scratch page (never pushed): the row-major hc tile, padding columns zeroed once
        cb_reserve_back(cb_hstage, 1);
        stage = reinterpret_cast<ptr_t>(get_write_ptr(cb_hstage));
        for (uint32_t w = 0; w < 1024; ++w) {
            stage[w] = 0;
        }
    }
#endif
    for (uint32_t i = 0; i < n_rooted; ++i) {
        const uint32_t r = get_arg_val<uint32_t>(8 + i);
        cb_wait_front(cb_fin, 1);
        noc_async_write_page(r, st_acc, get_read_ptr(cb_fin));
#ifdef MHC_POST_COLLAPSE
        cb_reserve_back(cb_hsoa, 1);
        cols_to_soa(reinterpret_cast<ptr_t>(get_read_ptr(cb_fin)), reinterpret_cast<ptr_t>(get_write_ptr(cb_hsoa)));
        cb_push_back(cb_hsoa, 1);
#endif
        noc_async_writes_flushed();
        cb_pop_front(cb_fin, 1);
#ifdef MHC_POST_COLLAPSE
        cb_wait_front(cb_hsoa_out, 1);
        noc_async_write_barrier();  // the previous row's hc write has left the stage
        soa_to_cols<hc_cols>(reinterpret_cast<ptr_t>(get_read_ptr(cb_hsoa_out)), stage);
        cb_pop_front(cb_hsoa_out, 1);
        noc_async_write_page(r, hc_acc, reinterpret_cast<uint32_t>(stage));
#endif
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
