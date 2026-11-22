// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// mhc_pre_xing_pack engram writer (BRISC, NoC 1). Per unit (mhc_engram_pack_common.hpp): the compute's three
// lane-wise accumulators (cb_acc: sum h w k, sum h^2, sum k^2) -> coefficient-major (cb_soa, lane = token row); their
// row totals (cb_soa_out, slot 0 of each) -> columns i, n + i, 2n + i of a zero row tile (cb_stage) -> scratch page
// r (n G) + bu, then one semaphore increment on row r's root core. Root phase: each rooted row's sum (cb_fin) -> the
// output page r. Word copies only (bit-exact).
// MHC_ENGRAM_GATE (option engram_gate_eps, P.24): the partial row tiles are laid coefficient-major (column k = SFPU
// vector k, lane = token row), so the root's sum is too and the compute's gate epilogue reads each column as one
// vector; the root's row is re-laid row-major (in the then free stage page) before its write.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "mhc_engram_pack_common.hpp"

namespace {

using mhc_engram::rc_index;
using mhc_engram::slot_index;
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

// MHC_ENGRAM_GATE: where the partial row keeps column `col` of token row l
inline uint32_t part_index(uint32_t l, uint32_t col) {
#ifdef MHC_ENGRAM_GATE
    return slot_index(col, l);
#else
    return rc_index(l, col);
#endif
}

}  // namespace

void kernel_main() {
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t ct = get_compile_time_arg_val(1);
    constexpr uint32_t rt = get_compile_time_arg_val(3);
    constexpr uint32_t bw = get_compile_time_arg_val(4);
    constexpr uint32_t groups = get_compile_time_arg_val(5);
    constexpr uint32_t page = get_compile_time_arg_val(6);
    constexpr uint32_t cb_acc = get_compile_time_arg_val(7);
    constexpr uint32_t cb_soa = get_compile_time_arg_val(8);
    constexpr uint32_t cb_soa_out = get_compile_time_arg_val(9);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(10);
    constexpr uint32_t cb_fin = get_compile_time_arg_val(11);
    constexpr uint32_t sem_id = get_compile_time_arg_val(12);
    constexpr auto s_args = TensorAccessorArgs<13>();
    constexpr auto o_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr uint32_t nu = n * groups;

    const auto s_acc = TensorAccessor(s_args, get_arg_val<uint32_t>(0), page);
    const auto o_acc = TensorAccessor(o_args, get_arg_val<uint32_t>(1), page);
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    const uint32_t n_rooted = get_arg_val<uint32_t>(4);
    constexpr uint32_t ROWS_ARG0 = 5;
    const uint32_t root_arg0 = ROWS_ARG0 + n_rooted;
    const uint32_t sem_addr = get_semaphore(sem_id);

    cb_reserve_back(cb_stage, 1);  // private scratch: the partial row tile, zero except the unit's 3 columns
    ptr_t stage = reinterpret_cast<ptr_t>(get_write_ptr(cb_stage));
    for (uint32_t w = 0; w < 1024; ++w) {
        stage[w] = 0;
    }

    for (uint32_t u = start; u < start + count; ++u) {
        const mhc_engram::Unit U = mhc_engram::unit_of(u, rt, groups, bw, ct);
        cb_wait_front(cb_acc, 3);
        cb_reserve_back(cb_soa, 3);
        for (uint32_t t = 0; t < 3; ++t) {
            cols_to_soa(
                reinterpret_cast<ptr_t>(get_read_ptr(cb_acc) + t * page),
                reinterpret_cast<ptr_t>(get_write_ptr(cb_soa) + t * page));
        }
        cb_push_back(cb_soa, 3);
        cb_pop_front(cb_acc, 3);
        cb_wait_front(cb_soa_out, 3);
        const uint32_t tot = get_read_ptr(cb_soa_out);
        for (uint32_t t = 0; t < 3; ++t) {
            ptr_t src = reinterpret_cast<ptr_t>(tot + t * page);
            const uint32_t col = t * n + U.i;
            for (uint32_t l = 0; l < 32; ++l) {
                stage[part_index(l, col)] = src[slot_index(0, l)];
            }
        }
        cb_pop_front(cb_soa_out, 3);
        noc_async_write_page(U.r * nu + U.bu, s_acc, reinterpret_cast<uint32_t>(stage));
        noc_async_write_barrier();
        for (uint32_t t = 0; t < 3; ++t) {
            const uint32_t col = t * n + U.i;
            for (uint32_t l = 0; l < 32; ++l) {
                stage[part_index(l, col)] = 0;
            }
        }
        const uint32_t xy = get_arg_val<uint32_t>(root_arg0 + (u - start));
        noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFFu, sem_addr), 1);
    }

    for (uint32_t j = 0; j < n_rooted; ++j) {
        const uint32_t r = get_arg_val<uint32_t>(ROWS_ARG0 + j);
        cb_wait_front(cb_fin, 1);
#ifdef MHC_ENGRAM_GATE
        {  // coefficient-major -> row-major in the stage page (free after the units)
            const volatile tt_l1_ptr uint32_t* fin = reinterpret_cast<ptr_t>(get_read_ptr(cb_fin));
            noc_async_writes_flushed();  // (the previous root row's write reads the stage page)
            for (uint32_t l = 0; l < 32; ++l) {
                for (uint32_t k = 0; k < 32; ++k) {
                    stage[rc_index(l, k)] = fin[slot_index(k, l)];
                }
            }
            noc_async_write_page(r, o_acc, reinterpret_cast<uint32_t>(stage));
        }
#else
        noc_async_write_page(r, o_acc, get_read_ptr(cb_fin));
#endif
        noc_async_writes_flushed();
        cb_pop_front(cb_fin, 1);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
