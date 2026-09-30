// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// mhc_pre_xing writer (BRISC, NoC 1): the layout moves of the coefficient stage and the output writes.
// Per segment (token tile-row r):
//   compute_coef: cb_row (row-major [mix 24 | sum x^2]) -> coefficient-major tile -> cb_soa; the finished
//                 coefficient-major [pre | post | comb] tile (cb_soa_out) -> the row-major hc stage (columns 24..31
//                 stay 0) -> hc DRAM page r (only the core whose segment starts at column 0 when with streams).
//   has_streams:  the pre-block tile (mhc_pre_xing_common.hpp) from the hc stage (or from the given hc row) ->
//                 cb_pb; then every y column tile of the segment cb_y -> y DRAM page r * Ct + c.
// pack_stats (per token tile-row r): the compute's lane-wise sum-of-squares tile (cb_acc, [token row, 32 column
//   partials]) -> coefficient-major (cb_soa, lane = token row); the compute's column total (cb_soa_out slot 0) ->
//   column n (n + 2) of the partial mix tile (cb_row) -> output page r.
// All moves are fp32 word copies (bit-exact).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "mhc_pre_layout.hpp"
#include "mhc_pre_xing_common.hpp"

namespace {

using mhc_layout::rc_index;

constexpr uint32_t soa_index(uint32_t k, uint32_t l) { return 64u * (k >> 1) + 2u * l + (k & 1u); }

// dst coefficient-major slots [0, COUNT) <- src row-major columns [0, COUNT)
template <uint32_t COUNT>
inline void cols_to_soa(const volatile tt_l1_ptr uint32_t* src, volatile tt_l1_ptr uint32_t* dst) {
#pragma GCC unroll 1
    for (uint32_t l = 0; l < 32; ++l) {
        const volatile tt_l1_ptr uint32_t* s = src + rc_index(l, 0);
        volatile tt_l1_ptr uint32_t* d = dst + 2u * l;
#pragma GCC unroll 32
        for (uint32_t k = 0; k < COUNT; ++k) {
            d[soa_index(k, 0)] = s[rc_index(0, k)];
        }
    }
}

// dst row-major columns [0, COUNT) <- src coefficient-major slots [0, COUNT)
template <uint32_t COUNT>
inline void soa_to_cols(const volatile tt_l1_ptr uint32_t* src, volatile tt_l1_ptr uint32_t* dst) {
#pragma GCC unroll 1
    for (uint32_t l = 0; l < 32; ++l) {
        const volatile tt_l1_ptr uint32_t* s = src + 2u * l;
        volatile tt_l1_ptr uint32_t* d = dst + rc_index(l, 0);
#pragma GCC unroll 32
        for (uint32_t k = 0; k < COUNT; ++k) {
            d[rc_index(0, k)] = s[soa_index(k, 0)];
        }
    }
}

// Pre-block tile: slot 8 i + b, lane l <- pre_i (row-major column i) of token row 4 b + (l >> 3).
template <uint32_t N>
inline void build_pre_blocks(const volatile tt_l1_ptr uint32_t* src, volatile tt_l1_ptr uint32_t* dst) {
#pragma GCC unroll 1
    for (uint32_t q = 0; q < 8 * N; ++q) {
        const uint32_t i = q >> 3, b = q & 7u;
        volatile tt_l1_ptr uint32_t* d = dst + soa_index(q, 0);
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

}  // namespace

void kernel_main() {
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t ct = get_compile_time_arg_val(1);
    constexpr bool compute_coef = get_compile_time_arg_val(2) != 0;
    constexpr bool has_streams = get_compile_time_arg_val(3) != 0;
    constexpr bool write_hc = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t cb_row = get_compile_time_arg_val(5);
    constexpr uint32_t cb_soa = get_compile_time_arg_val(6);
    constexpr uint32_t cb_soa_out = get_compile_time_arg_val(7);
    constexpr uint32_t cb_hc_stage = get_compile_time_arg_val(8);
    constexpr uint32_t cb_pb = get_compile_time_arg_val(9);
    constexpr uint32_t cb_y = get_compile_time_arg_val(10);
    constexpr uint32_t hc_page = get_compile_time_arg_val(11);
    constexpr uint32_t y_page = get_compile_time_arg_val(12);
    constexpr uint32_t mix_slots = get_compile_time_arg_val(13);  // n (n + 2) + 1: mixes + sum x^2
    constexpr uint32_t coef_cols = get_compile_time_arg_val(14);  // n (n + 2)
    constexpr bool pack_stats = get_compile_time_arg_val(15) != 0;
    constexpr uint32_t cb_acc = get_compile_time_arg_val(16);
    constexpr auto hc_args = TensorAccessorArgs<17>();
    constexpr auto y_args = TensorAccessorArgs<hc_args.next_compile_time_args_offset()>();
    static_assert(n * 8 <= 32, "the pre-block tile holds 8 vectors per stream");

    const uint32_t hc_addr = get_arg_val<uint32_t>(0);
    const uint32_t y_addr = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);

    const auto hc_acc = TensorAccessor(hc_args, hc_addr, hc_page);
    const auto y_acc = TensorAccessor(y_args, y_addr, y_page);

    using ptr_t = volatile tt_l1_ptr uint32_t*;
    if constexpr (pack_stats) {
        for (uint32_t r = start; r < start + count; ++r) {
            cb_wait_front(cb_acc, 1);
            cb_reserve_back(cb_soa, 1);
            cols_to_soa<32>(
                reinterpret_cast<ptr_t>(get_read_ptr(cb_acc)), reinterpret_cast<ptr_t>(get_write_ptr(cb_soa)));
            cb_push_back(cb_soa, 1);
            cb_pop_front(cb_acc, 1);
            cb_wait_front(cb_soa_out, 1);
            cb_wait_front(cb_row, 1);
            ptr_t tot = reinterpret_cast<ptr_t>(get_read_ptr(cb_soa_out));
            ptr_t row = reinterpret_cast<ptr_t>(get_read_ptr(cb_row));
            for (uint32_t l = 0; l < 32; ++l) {
                row[rc_index(l, coef_cols)] = tot[soa_index(0, l)];
            }
            cb_pop_front(cb_soa_out, 1);
            noc_async_write_page(r, hc_acc, get_read_ptr(cb_row));
            noc_async_writes_flushed();
            cb_pop_front(cb_row, 1);
        }
        noc_async_write_barrier();
        return;
    }
    ptr_t stage = nullptr;
    if constexpr (compute_coef) {
        // Private scratch page (never pushed): the row-major hc tile, padding columns zeroed once.
        cb_reserve_back(cb_hc_stage, 1);
        stage = reinterpret_cast<ptr_t>(get_write_ptr(cb_hc_stage));
        for (uint32_t w = 0; w < 1024; ++w) {
            stage[w] = 0;
        }
    }

    mhc_xing::SegmentWalker walker(start, count, has_streams ? ct : 1);
    while (!walker.done()) {
        const mhc_xing::Segment seg = walker.next();
        cb_wait_front(cb_row, 1);
        ptr_t row = reinterpret_cast<ptr_t>(get_read_ptr(cb_row));
        ptr_t pre_src = row;
        if constexpr (compute_coef) {
            cb_reserve_back(cb_soa, 1);
            cols_to_soa<mix_slots>(row, reinterpret_cast<ptr_t>(get_write_ptr(cb_soa)));
            cb_push_back(cb_soa, 1);
            cb_pop_front(cb_row, 1);
            cb_wait_front(cb_soa_out, 1);
            noc_async_write_barrier();  // the previous row's hc write has left the stage
            soa_to_cols<coef_cols>(reinterpret_cast<ptr_t>(get_read_ptr(cb_soa_out)), stage);
            cb_pop_front(cb_soa_out, 1);
            if constexpr (write_hc) {
                if (!has_streams || seg.col0 == 0) {
                    noc_async_write_page(seg.row, hc_acc, reinterpret_cast<uint32_t>(stage));
                }
            }
            pre_src = stage;
        }
        if constexpr (has_streams) {
            cb_reserve_back(cb_pb, 1);
            build_pre_blocks<n>(pre_src, reinterpret_cast<ptr_t>(get_write_ptr(cb_pb)));
            cb_push_back(cb_pb, 1);
            if constexpr (!compute_coef) {
                cb_pop_front(cb_row, 1);
            }
            const uint32_t y_base = seg.row * ct;
            for (uint32_t c = seg.col0; c < seg.col0 + seg.cols; ++c) {
                cb_wait_front(cb_y, 1);
                noc_async_write_page(y_base + c, y_acc, get_read_ptr(cb_y));
                noc_async_writes_flushed();
                cb_pop_front(cb_y, 1);
            }
        }
    }
    noc_async_write_barrier();
}
