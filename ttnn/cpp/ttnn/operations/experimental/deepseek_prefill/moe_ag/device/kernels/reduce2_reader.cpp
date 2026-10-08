// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// All-gather MoE local reduce in two phases (two mesh rows), reader (NCRISC). PHASE 1: the other mesh row's tokens;
// PHASE 2: this row's tokens, each with one extra pair (the peer's phase-1 partial for it, gathered, weight 1), so the
// send-back needs no separate add. For each token g of this core's range (g0 + the block start from the chip info:
// word 1 the other row's block, word 2 this row's): its local expert outputs (y_slot[g, k] != none) as (row-major bf16
// y row, weight w[g, k]) pairs; a phase-1 token with no local expert gets one (zero row, 0) pair. Per token: a header
// page (the pair count, word 0), then per pair TILES tiles of 1024 elements (c_0) and a scalar tile (c_1, element 0 =
// the bf16 weight).
// CT: 0 K, 1 ROW_BYTES, 2 TILES (2 KB blocks per staged row, ceil(ROW_BYTES / 2048)), 3 W_STRIDE (L1 bytes per staged
// weight page), 4 PHASE Common RT: 0 y (row-major bf16 [rows, H]) addr, 1 y_slot addr, 2 w (gathered weights [T, K]
// bf16) addr, 3 S, 4 tokens
//     per core (range g0, n within the block), 5 chip-info addr, 6 peer partials (phase 2: the gathered phase-1
//     partials [2 S, H]) addr, 7 grid x
// CBs: 0 y rows, 1 weight tiles, 2 headers, 4 y_slot block (scratch), 5 weights (scratch), 6 zero row (scratch)
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "core_range.hpp"

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(1);
    constexpr uint32_t TILES = get_compile_time_arg_val(2);
    constexpr uint32_t W_STRIDE = get_compile_time_arg_val(3);
    constexpr uint32_t PHASE = get_compile_time_arg_val(4);
    constexpr uint32_t NONE = 0xFFFFFFFFu;
    constexpr uint32_t cb_y = tt::CBIndex::c_0, cb_w = tt::CBIndex::c_1, cb_h = tt::CBIndex::c_2;
    const uint32_t y_addr = get_common_arg_val<uint32_t>(0), ys_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(2);
    const auto [g0_, n] =
        core_range(get_common_arg_val<uint32_t>(3), get_common_arg_val<uint32_t>(4), get_common_arg_val<uint32_t>(7));
    const InterleavedAddrGen<true> pg = {.bank_base_address = get_common_arg_val<uint32_t>(6), .page_size = ROW_BYTES};
    uint32_t g0 = g0_, peer0 = 0;
    {
        const uint32_t l1 = get_write_ptr(tt::CBIndex::c_7);
        noc_async_read(
            get_noc_addr(
                0, InterleavedAddrGen<true>{.bank_base_address = get_common_arg_val<uint32_t>(5), .page_size = 64}),
            l1,
            64);
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* info = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
        peer0 = info[1] + g0;  // the peer's partial for this row's token i sits at the other block's row i
        g0 += PHASE == 1 ? info[1] : info[2];
    }
    const InterleavedAddrGen<true> yg = {.bank_base_address = y_addr, .page_size = ROW_BYTES};
    const InterleavedAddrGen<true> wg = {.bank_base_address = w_addr, .page_size = K * 2};
    const uint32_t l1_ys = get_write_ptr(tt::CBIndex::c_4), l1_w = get_write_ptr(tt::CBIndex::c_5);
    const uint32_t l1_zero = get_write_ptr(tt::CBIndex::c_6);
    if (n == 0) {
        return;
    }
    noc_async_read(
        get_noc_addr(0, InterleavedAddrGen<true>{.bank_base_address = ys_addr, .page_size = 4}) + g0 * K * 4,
        l1_ys,
        n * K * 4);
    for (uint32_t i = 0; i < n; ++i) {
        noc_async_read(get_noc_addr(g0 + i, wg), l1_w + i * W_STRIDE, K * 2);
    }
    {
        tt_l1_ptr uint32_t* z = reinterpret_cast<tt_l1_ptr uint32_t*>(l1_zero);
        for (uint32_t i = 0; i < ROW_BYTES / 4; ++i) {
            z[i] = 0;
        }
    }
    noc_async_read_barrier();
    const tt_l1_ptr uint32_t* ys = reinterpret_cast<const tt_l1_ptr uint32_t*>(l1_ys);
    const uint64_t zero_src = get_noc_addr(l1_zero);
    for (uint32_t i = 0; i < n; ++i) {
        const tt_l1_ptr uint16_t* w = reinterpret_cast<const tt_l1_ptr uint16_t*>(l1_w + i * W_STRIDE);
        uint32_t cnt = 0;
        for (uint32_t k = 0; k < K; ++k) {
            cnt += ys[i * K + k] != NONE;
        }
        const bool extra = PHASE == 2;
        cb_reserve_back(cb_h, 1);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_h)) = extra ? cnt + 1 : (cnt ? cnt : 1);
        cb_push_back(cb_h, 1);
        if (extra) {
            cb_reserve_back(cb_y, TILES);
            cb_reserve_back(cb_w, 1);
            noc_async_read(get_noc_addr(peer0 + i, pg), get_write_ptr(cb_y), ROW_BYTES);
            *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_w)) = 0x3F80;  // bf16 1.0
            noc_async_read_barrier();
            cb_push_back(cb_w, 1);
            cb_push_back(cb_y, TILES);
        }
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t row = ys[i * K + k];
            if (row == NONE && (extra || !(cnt == 0 && k == K - 1))) {
                continue;
            }
            cb_reserve_back(cb_y, TILES);
            cb_reserve_back(cb_w, 1);
            noc_async_read(row == NONE ? zero_src : get_noc_addr(row, yg), get_write_ptr(cb_y), ROW_BYTES);
            *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_w)) = row == NONE ? 0 : w[k];
            noc_async_read_barrier();
            cb_push_back(cb_w, 1);
            cb_push_back(cb_y, TILES);
        }
    }
}
