// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// FlashFused reader: feeds the production sdpa_standard() compute. Reads this core's Q head (one
// seq-tile chunk, DHt tiles) once, then the single MQA KV head chunk-by-chunk (Sk_chunk_t*DHt tiles).
//
// KV source: tiles [0, prefix_Kt) come from the resident prefix (past_k/past_v) and the rest from the
// new/suffix K/V (k/v) -- so the caller does not pre-concatenate. When has_past == 0, prefix_Kt == 0,
// the prefix accessors alias k/v and are never read, and all KV tiles come from k/v.
void kernel_main() {
    constexpr uint32_t NQH = get_compile_time_arg_val(0);
    constexpr uint32_t DHt = get_compile_time_arg_val(1);
    constexpr uint32_t Kt = get_compile_time_arg_val(2);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(4);
    constexpr uint32_t prefix_Kt = get_compile_time_arg_val(5);
    constexpr bool has_past = get_compile_time_arg_val(6) == 1;
    constexpr uint32_t suffix_Kt = Kt - prefix_Kt;
    constexpr auto q_args = TensorAccessorArgs<7>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto pk_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto pv_args = TensorAccessorArgs<pk_args.next_compile_time_args_offset()>();

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t q_head = get_arg_val<uint32_t>(3);
    const uint32_t kv_head = get_arg_val<uint32_t>(4);
    const uint32_t pk_addr = get_arg_val<uint32_t>(5);
    const uint32_t pv_addr = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;

    const uint32_t q_tb = get_tile_size(cb_q_in);
    const uint32_t k_tb = get_tile_size(cb_k_in);
    const uint32_t v_tb = get_tile_size(cb_v_in);
    const auto q_acc = TensorAccessor(q_args, q_addr, q_tb);
    const auto k_acc = TensorAccessor(k_args, k_addr, k_tb);
    const auto v_acc = TensorAccessor(v_args, v_addr, v_tb);
    const auto pk_acc = TensorAccessor(pk_args, pk_addr, k_tb);
    const auto pv_acc = TensorAccessor(pv_args, pv_addr, v_tb);

    // Q: head q_head, single seq-tile chunk -> DHt tiles. page = q_head*DHt + d.
    cb_reserve_back(cb_q_in, DHt);
    uint32_t l1 = get_write_ptr(cb_q_in);
    for (uint32_t d = 0; d < DHt; ++d) {
        noc_async_read_tile(q_head * DHt + d, q_acc, l1);
        l1 += q_tb;
    }
    noc_async_read_barrier();
    cb_push_back(cb_q_in, DHt);

    // K/V chunk by chunk. Per KV tile g: prefix tiles [0,prefix_Kt) from past_*, suffix from k/v.
    // Page within a source = (kv_head * src_Kt + local) * DHt + d.
    for (uint32_t c = 0; c < k_num_chunks; ++c) {
        const uint32_t kt0 = c * Sk_chunk_t;
        cb_reserve_back(cb_k_in, Sk_chunk_t * DHt);
        uint32_t lk = get_write_ptr(cb_k_in);
        cb_reserve_back(cb_v_in, Sk_chunk_t * DHt);
        uint32_t lv = get_write_ptr(cb_v_in);
        for (uint32_t kt = 0; kt < Sk_chunk_t; ++kt) {
            const uint32_t g = kt0 + kt;
            if (has_past && g < prefix_Kt) {
                const uint32_t base = (kv_head * prefix_Kt + g) * DHt;
                for (uint32_t d = 0; d < DHt; ++d) {
                    noc_async_read_tile(base + d, pk_acc, lk + d * k_tb);
                    noc_async_read_tile(base + d, pv_acc, lv + d * v_tb);
                }
            } else {
                const uint32_t base = (kv_head * suffix_Kt + (g - prefix_Kt)) * DHt;
                for (uint32_t d = 0; d < DHt; ++d) {
                    noc_async_read_tile(base + d, k_acc, lk + d * k_tb);
                    noc_async_read_tile(base + d, v_acc, lv + d * v_tb);
                }
            }
            lk += DHt * k_tb;
            lv += DHt * v_tb;
        }
        noc_async_read_barrier();
        cb_push_back(cb_k_in, Sk_chunk_t * DHt);
        cb_push_back(cb_v_in, Sk_chunk_t * DHt);
    }
    (void)NQH;
}
