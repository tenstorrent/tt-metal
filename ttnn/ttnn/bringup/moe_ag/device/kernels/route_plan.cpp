// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// All-gather MoE: local route plan (BRISC, R = 64 cores in one 8 x 8 rectangle). From the dispatch group's gathered
// top-k indices [T, K] uint16 (every chip of a mesh column holds the same T tokens) and this chip's local-slot map
// (global expert id -> local expert slot, or >= EPC), builds on device:
//   counts  [1, NG] uint32: tokens per global expert (this chip's experts; 0 elsewhere)
//   regions [1, NG] uint32: each local expert's first row in the flat expert space (local order, 32-row aligned)
//   token_index [1, BUF] uint32: token_index[region_e + i] = gathered row of expert e's i-th token (ascending g),
//       the tile tail of each region zero
//   y_slot [1, T * K] uint32: per (token, k) the flat row of its expert output, 0xFFFFFFFF if not a local expert
// Core r owns the token range [g0, g0 + n) and, if r < EPC, local expert r. Every loop is O(R + EPC + n K + NG):
//   A  range histogram; hist[r][e] inline-written into expert core e's column (L1)       -> core 0 (S_A), go 1
//   C1 expert core e: prefix of its column over ranges -> start row of each range inline-written into range core r's
//      message, its count inline-written to core 0                                     -> core 0 (S_C)
//   C2 core 0: regions (prefix of padded counts) -> multicast region | count per expert, the counts / regions rows, go
//   2 B  rescan: y_slot of the range; each local pair's gathered row inline-written into the expert core's list at its
//      rank                                                                             -> core 0 (S_E), go 3
//   D  expert core: zero its list's tile tail, write the list at its region.
// Go signals: core 0 multicasts a 1 into every core's go semaphore; each core clears its own after the wait.
// CT: 0 R, 1 EPC, 2 NG, 3 K, 4 IDX_STRIDE (L1 bytes per staged idx page), 5-8 the rectangle's NoC corners
// Common RT: 0 idx addr, 1 lmap addr, 2 counts addr, 3 regions addr, 4 token_index addr, 5 y_slot addr, 6 T, 7 tokens
//     per range, 8 (unused), 9.. the R cores' NoC xy (x << 16 | y), core 0 first. Core me = y * 8 + x (logical)
// CBs (scratch): 0 idx pages, 1 lmap, 2 column (expert core: hist[r][me]), 3 message (start | region | count),
//     4 y_slot block, 5 expert list, 6 core 0: counts / regions rows, 7 core 0: counts per local expert + a 1 word
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "core_range.hpp"

void kernel_main() {
    constexpr uint32_t R = get_compile_time_arg_val(0);
    constexpr uint32_t EPC = get_compile_time_arg_val(1);
    constexpr uint32_t NG = get_compile_time_arg_val(2);
    constexpr uint32_t K = get_compile_time_arg_val(3);
    constexpr uint32_t IDX_STRIDE = get_compile_time_arg_val(4);
    constexpr uint32_t MC_X0 = get_compile_time_arg_val(5), MC_Y0 = get_compile_time_arg_val(6);
    constexpr uint32_t MC_X1 = get_compile_time_arg_val(7), MC_Y1 = get_compile_time_arg_val(8);
    constexpr uint32_t S_A = 0, S_C = 1, S_E = 2, G1 = 3, G2 = 4, G3 = 5;
    constexpr uint32_t NONE = 0xFFFFFFFFu;

    const uint32_t idx_addr = get_common_arg_val<uint32_t>(0), lmap_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t counts_addr = get_common_arg_val<uint32_t>(2), regions_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t tidx_addr = get_common_arg_val<uint32_t>(4), yslot_addr = get_common_arg_val<uint32_t>(5);
    const uint32_t me = core_index(8);
    const auto [g0, n] = core_range(get_common_arg_val<uint32_t>(6), get_common_arg_val<uint32_t>(7), 8);
    auto noc = [](uint32_t r, uint32_t addr) {
        const uint32_t v = get_common_arg_val<uint32_t>(9 + r);
        return get_noc_addr(v >> 16, v & 0xFFFF, addr);
    };
    auto sem = [](uint32_t id) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id)); };
    auto dram = [](uint32_t addr) {
        return get_noc_addr(0, InterleavedAddrGen<true>{.bank_base_address = addr, .page_size = 4});
    };

    const uint32_t l1_idx = get_write_ptr(tt::CBIndex::c_0), l1_lmap = get_write_ptr(tt::CBIndex::c_1);
    const uint32_t l1_col = get_write_ptr(tt::CBIndex::c_2), l1_msg = get_write_ptr(tt::CBIndex::c_3);
    const uint32_t l1_ys = get_write_ptr(tt::CBIndex::c_4), l1_list = get_write_ptr(tt::CBIndex::c_5);
    const uint32_t l1_rows = get_write_ptr(tt::CBIndex::c_6), l1_cnt = get_write_ptr(tt::CBIndex::c_7);
    auto go = [&](uint32_t id) {  // core 0: a 1 into every core's go semaphore
        const uint32_t one = l1_cnt + EPC * 4;
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(one) = 1;
        noc_semaphore_set_multicast_loopback_src(
            one, get_noc_multicast_addr(MC_X0, MC_Y0, MC_X1, MC_Y1, get_semaphore(id)), R);
        noc_async_write_barrier();
    };
    auto wait_go = [&](uint32_t id) {
        noc_semaphore_wait(sem(id), 1);
        noc_semaphore_set(sem(id), 0);
    };
    auto join = [&](uint32_t id, uint32_t count) {  // core 0 waits for count arrivals, clears
        noc_semaphore_wait(sem(id), count);
        noc_semaphore_set(sem(id), 0);
    };

    // inputs: the range's idx pages (one token per page), the local-slot map
    const InterleavedAddrGen<true> idx_g = {.bank_base_address = idx_addr, .page_size = K * 2};
    for (uint32_t i = 0; i < n; ++i) {
        noc_async_read(get_noc_addr(g0 + i, idx_g), l1_idx + i * IDX_STRIDE, K * 2);
    }
    noc_async_read(dram(lmap_addr), l1_lmap, NG * 4);
    noc_async_read_barrier();
    const tt_l1_ptr uint32_t* lmap = reinterpret_cast<const tt_l1_ptr uint32_t*>(l1_lmap);
    auto slot = [&](uint32_t i, uint32_t k) -> uint32_t {
        const uint32_t gid = reinterpret_cast<const tt_l1_ptr uint16_t*>(l1_idx + i * IDX_STRIDE)[k];
        return gid < NG ? lmap[gid] : NONE;
    };

    // A: histogram of the range -> column r of every expert core
    volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_msg);  // start|region|count
    uint32_t hist[EPC];
    for (uint32_t e = 0; e < EPC; ++e) {
        hist[e] = 0;
    }
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t l = slot(i, k);
            if (l < EPC) {
                hist[l] += 1;
            }
        }
    }
    for (uint32_t e = 0; e < EPC; ++e) {
        noc_inline_dw_write(noc(e, l1_col + me * 4), hist[e]);
    }
    noc_async_write_barrier();
    noc_semaphore_inc(noc(0, get_semaphore(S_A)), 1);
    if (me == 0) {
        join(S_A, R);
        go(G1);
    }

    // C1: expert core: its column's prefix -> each range's start row; its count -> core 0
    wait_go(G1);
    if (me < EPC) {
        volatile tt_l1_ptr uint32_t* col = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_col);
        uint32_t run = 0;
        for (uint32_t r = 0; r < R; ++r) {
            const uint32_t h = col[r];
            noc_inline_dw_write(noc(r, l1_msg + me * 4), run);
            run += h;
        }
        noc_inline_dw_write(noc(0, l1_cnt + me * 4), run);
        noc_async_write_barrier();
        noc_semaphore_inc(noc(0, get_semaphore(S_C)), 1);
    }
    // C2: core 0: regions, the region | count table to every core, the rows
    if (me == 0) {
        join(S_C, EPC);
        volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_cnt);
        uint32_t region = 0;
        for (uint32_t e = 0; e < EPC; ++e) {
            const uint32_t c = cnt[e];
            m[EPC + e] = region;
            m[2 * EPC + e] = c;
            region += (c + 31) & ~31u;
        }
        noc_async_write_multicast_loopback_src(
            l1_msg + EPC * 4,
            get_noc_multicast_addr(MC_X0, MC_Y0, MC_X1, MC_Y1, l1_msg + EPC * 4),
            2 * EPC * 4,
            R,
            false);
        tt_l1_ptr uint32_t* rows = reinterpret_cast<tt_l1_ptr uint32_t*>(l1_rows);
        for (uint32_t gid = 0; gid < NG; ++gid) {
            const uint32_t l = lmap[gid];
            rows[gid] = l < EPC ? m[2 * EPC + l] : 0;
            rows[NG + gid] = l < EPC ? m[EPC + l] : 0;
        }
        noc_async_write(l1_rows, dram(counts_addr), NG * 4);
        noc_async_write(l1_rows + NG * 4, dram(regions_addr), NG * 4);
        noc_async_write_barrier();
        go(G2);
    }

    // B: y_slot of the range; the pairs' gathered rows into the expert cores' lists
    wait_go(G2);
    {
        uint32_t start[EPC], region[EPC];
        for (uint32_t e = 0; e < EPC; ++e) {
            start[e] = m[e];
            region[e] = m[EPC + e];
        }
        tt_l1_ptr uint32_t* ys = reinterpret_cast<tt_l1_ptr uint32_t*>(l1_ys);
        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t k = 0; k < K; ++k) {
                const uint32_t l = slot(i, k);
                if (l < EPC) {
                    const uint32_t rel = start[l]++;
                    ys[i * K + k] = region[l] + rel;
                    noc_inline_dw_write(noc(l, l1_list + rel * 4), g0 + i);
                } else {
                    ys[i * K + k] = NONE;
                }
            }
        }
        if (n) {
            noc_async_write(l1_ys, dram(yslot_addr) + g0 * K * 4, n * K * 4);
        }
    }
    noc_async_write_barrier();
    noc_semaphore_inc(noc(0, get_semaphore(S_E)), 1);
    if (me == 0) {
        join(S_E, R);
        go(G3);
    }

    // D: the expert core writes its list (tile tail zeroed) at its region
    wait_go(G3);
    if (me < EPC) {
        const uint32_t count = m[2 * EPC + me], region = m[EPC + me], padded = (count + 31) & ~31u;
        volatile tt_l1_ptr uint32_t* list = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_list);
        for (uint32_t i = count; i < padded; ++i) {
            list[i] = 0;
        }
        if (padded) {
            noc_async_write(l1_list, dram(tidx_addr) + region * 4, padded * 4);
        }
    }
    noc_async_full_barrier();
}
