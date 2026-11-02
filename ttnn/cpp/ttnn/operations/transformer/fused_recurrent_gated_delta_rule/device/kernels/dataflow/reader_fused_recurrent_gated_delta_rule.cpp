// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Reader: initial state S [K,V] once (from s0 or host-provided zeros), then per token
// q,k [1,K], v [1,V], decay,beta [1,1]. All fp32. Device 2.0 API.
// Per-token layout: q/k/v are [BH*T, 1, D] and decay/beta [BH*T, 1, 1]; block index = h*T + t.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/debug/assert.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_decay = 3, cb_beta = 4, cb_S = 5;
constexpr uint32_t cb_blkidx = 16;  // ring mode only: scratch for the [BH] block-index vector
constexpr uint32_t cb_gbscr = 17;   // gb_fused only: one-tile scratch for the a|b / constants rows

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t use_blk_idx = get_compile_time_arg_val(2);
    constexpr uint32_t gb_fused = get_compile_time_arg_val(3);
    // Vt is this core's share of V (all of it, or half under QWEN36_1C_GDN_VSPLIT); Vt_full addresses the tensors.
    constexpr uint32_t Vt_full = get_compile_time_arg_val(4);

    constexpr auto q_a = TensorAccessorArgs<5>();
    constexpr auto k_a = TensorAccessorArgs<q_a.next_compile_time_args_offset()>();
    constexpr auto v_a = TensorAccessorArgs<k_a.next_compile_time_args_offset()>();
    constexpr auto d_a = TensorAccessorArgs<v_a.next_compile_time_args_offset()>();
    constexpr auto b_a = TensorAccessorArgs<d_a.next_compile_time_args_offset()>();
    constexpr auto s0_a = TensorAccessorArgs<b_a.next_compile_time_args_offset()>();
    constexpr auto idx_a = TensorAccessorArgs<s0_a.next_compile_time_args_offset()>();

    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t T = get_arg_val<uint32_t>(1);
    const uint32_t q_addr = get_arg_val<uint32_t>(2);
    const uint32_t k_addr = get_arg_val<uint32_t>(3);
    const uint32_t v_addr = get_arg_val<uint32_t>(4);
    const uint32_t d_addr = get_arg_val<uint32_t>(5);
    const uint32_t b_addr = get_arg_val<uint32_t>(6);
    const uint32_t s0_addr = get_arg_val<uint32_t>(7);
    // rt 8 = block-index buffer address (0 when absent), rt 9 = its aligned page size, rt 10 = BH.

    const uint32_t tb = get_tile_size(cb_q);  // fp32; all tensors share it
    const auto q_acc = TensorAccessor(q_a, q_addr, tb);
    const auto k_acc = TensorAccessor(k_a, k_addr, tb);
    const auto v_acc = TensorAccessor(v_a, v_addr, tb);
    const auto d_acc = TensorAccessor(d_a, d_addr, tb);
    const auto b_acc = TensorAccessor(b_a, b_addr, tb);
    const auto s0_acc = TensorAccessor(s0_a, s0_addr, tb);

    constexpr uint32_t kv = Kt * Vt;

    Noc noc;
    auto read_into = [&](const auto& acc, uint32_t cb_id, uint32_t base, uint32_t n) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(n);
        for (uint32_t t = 0; t < n; t++) {
            noc.async_read(acc, cb, tb, {.page_id = base + t}, {.offset_bytes = t * tb});
        }
        noc.async_read_barrier();
        cb.push_back(n);
    };

    // gb_fused (T == 1): `decay` is the raw a|b row [1,1,2BH] and `beta` the constants row [1,1,2BH]
    // (-exp(A_log) | dt_bias). Pick this head's four scalars out of row 0 (column c of a tile lives in face c/16
    // at index c%16) and hand them to compute as zero tiles with the value at [0,0]: a, dt_bias, -exp(A_log) in
    // cb_decay and b in cb_beta. Every other element must be finite (zero): the compute kernel scales row-0 data
    // by beta's column 0 and runs the padding rows through the rank-1 matmul, so garbage there would reach S.
    // Done first, since the compute kernel's first step needs decay.
    if constexpr (gb_fused) {
        const uint32_t BH = get_arg_val<uint32_t>(10);
        // The four source tiles go to four scratch slots under one barrier (was one round trip each).
        CircularBuffer scr(cb_gbscr);
        const CoreLocalMem<volatile uint32_t> sp(scr.get_write_ptr());
        constexpr uint32_t words = 1024;  // fp32 32x32 tile
        const uint32_t cols[4] = {h, BH + h, h, BH + h};  // a, b (a|b row); -exp(A_log), dt_bias (constants row)
        for (uint32_t i = 0; i < 4; i++) {
            noc.async_read(i < 2 ? d_acc : b_acc, scr, tb, {.page_id = cols[i] / 32}, {.offset_bytes = i * tb});
        }
        noc.async_read_barrier();
        auto elem = [&](uint32_t i) -> uint32_t {
            const uint32_t c = cols[i] % 32;
            return sp[i * words + (c / 16) * 256 + (c % 16)];
        };
        const uint32_t a = elem(0), b = elem(1), neg_a = elem(2), dt = elem(3);
        // Only what the compute kernel reads must be set: decay is consumed as a broadcast scalar ([0,0]) and its
        // raw inputs are elementwise, so their [0,0] alone matters; beta is consumed down its COLUMN 0 (one scale per
        // row, and rows 1-31 meet the zero padding of v - vread), so column 0 of the raw b tile must be finite.
        // Zeroing those 32 words instead of whole tiles saves ~4000 RISC stores per call.
        CircularBuffer cd(cb_decay), cbb(cb_beta);
        cd.reserve_back(3);
        const CoreLocalMem<volatile uint32_t> wd(cd.get_write_ptr());
        wd[0] = a;
        wd[words] = dt;
        wd[2 * words] = neg_a;
        cd.push_back(3);
        cbb.reserve_back(1);
        const CoreLocalMem<volatile uint32_t> wb(cbb.get_write_ptr());
        for (uint32_t r = 0; r < 16; r++) {
            wb[r * 16] = 0;        // face 0, column 0, rows 0-15
            wb[512 + r * 16] = 0;  // face 2, column 0, rows 16-31
        }
        wb[0] = b;
        cbb.push_back(1);
    }

    // initial state S (once) — host always provides it (zeros if absent). In ring mode the block
    // is idx[h] instead of h (caller contract: see the Python binding).
    uint32_t s0_block = h;
    if constexpr (use_blk_idx) {
        const uint32_t idx_addr = get_arg_val<uint32_t>(8);
        const uint32_t idx_page_bytes = get_arg_val<uint32_t>(9);
        [[maybe_unused]] const uint32_t BH = get_arg_val<uint32_t>(10);
        CircularBuffer cb_idx(cb_blkidx);
        const CoreLocalMem<volatile uint32_t> idx(cb_idx.get_write_ptr());
        const auto idx_acc = TensorAccessor(idx_a, idx_addr, idx_page_bytes);
        noc.async_read(idx_acc, cb_idx, idx_page_bytes, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        s0_block = idx[h];
        // Watcher-checked caller contract: the block belongs to head h and lies inside the ring.
        ASSERT(s0_block % BH == h && s0_block < T * BH);
    }
    // This core's V columns of S: Kt row segments of Vt tiles, starting at tile v0 of each row of Vt_full.
    const uint32_t v0 = get_arg_val<uint32_t>(12);
    if constexpr (Vt == Vt_full) {
        read_into(s0_acc, cb_S, s0_block * kv, kv);
    } else {
        CircularBuffer cbS(cb_S);
        cbS.reserve_back(kv);
        for (uint32_t kt = 0; kt < Kt; kt++) {
            for (uint32_t vt = 0; vt < Vt; vt++) {
                noc.async_read(s0_acc, cbS, tb, {.page_id = s0_block * Kt * Vt_full + kt * Vt_full + v0 + vt},
                               {.offset_bytes = (kt * Vt + vt) * tb});
            }
        }
        noc.async_read_barrier();
        cbS.push_back(kv);
    }

    // GQA: q/k may carry only H = HV/gqa query heads, so value-head h reads query head h/gqa. With
    // gqa == 1 this is exactly h*T+t, the pre-expanded form, so the pre-expanded callers are
    // unaffected. Doing the replicate here costs one divide per core instead of a
    // repeat_interleave tensor on the host (two untilize/concat/tilize/copy chains at decode).
    const uint32_t gqa = get_arg_val<uint32_t>(11);
    const uint32_t qk_h = h / gqa;

    for (uint32_t t = 0; t < T; t++) {
        const uint32_t block = h * T + t;
        const uint32_t qk_block = qk_h * T + t;
        read_into(q_acc, cb_q, qk_block * Kt, Kt);
        read_into(k_acc, cb_k, qk_block * Kt, Kt);
        read_into(v_acc, cb_v, block * Vt_full + v0, Vt);
        if constexpr (!gb_fused) {
            read_into(d_acc, cb_decay, block, 1);
            read_into(b_acc, cb_beta, block, 1);
        }
    }
}
