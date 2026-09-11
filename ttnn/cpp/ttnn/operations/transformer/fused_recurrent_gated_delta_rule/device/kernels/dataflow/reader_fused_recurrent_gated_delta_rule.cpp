// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Reader: initial state S [K,V] once (from s0 or host-provided zeros), then per token
// q,k [1,K], v [1,V], decay,beta [1,1]. All fp32. Device 2.0 API.
// Per-token layout: q/k/v are [BH*T, 1, D] and decay/beta [BH*T, 1, 1]; block index = h*T + t.
//
// Ring mode (use_blk_idx): s0 is the [BH*T,K,V] per-token-state ring the writer also writes into,
// and this core's initial state is block idx[h] of it, not block h. idx is one ROW_MAJOR page of
// BH uint32s; read it whole into a scratch CB and take element h. The caller guarantees
// idx[h] % BH == h, so this core only reads a block it wrote itself, and the read completes
// (barrier + push of cb_S) before compute produces the first state the writer stores -- no
// cross-core ordering and no semaphore.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_decay = 3, cb_beta = 4, cb_S = 5;
constexpr uint32_t cb_blkidx = 16;  // ring mode only: scratch for the [BH] block-index vector

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t use_blk_idx = get_compile_time_arg_val(2);

    constexpr auto q_a = TensorAccessorArgs<3>();
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
    // rt 8 = idx buffer address (0 when absent), rt 9 = its aligned page size; read below.

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

    // initial state S (once) — host always provides it (zeros if absent). In ring mode the block
    // is selected per head by idx[h] instead of being h.
    uint32_t s0_block = h;
    if constexpr (use_blk_idx) {
        const uint32_t idx_addr = get_arg_val<uint32_t>(8);
        const uint32_t idx_page_bytes = get_arg_val<uint32_t>(9);
        CircularBuffer cb_idx(cb_blkidx);
        const uint32_t idx_l1 = cb_idx.get_write_ptr();
        const auto idx_acc = TensorAccessor(idx_a, idx_addr, idx_page_bytes);
        noc.async_read(idx_acc, cb_idx, idx_page_bytes, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        s0_block = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1)[h];
    }
    read_into(s0_acc, cb_S, s0_block * kv, kv);

    for (uint32_t t = 0; t < T; t++) {
        const uint32_t block = h * T + t;
        read_into(q_acc, cb_q, block * Kt, Kt);
        read_into(k_acc, cb_k, block * Kt, Kt);
        read_into(v_acc, cb_v, block * Vt, Vt);
        read_into(d_acc, cb_decay, block, 1);
        read_into(b_acc, cb_beta, block, 1);
    }
}
