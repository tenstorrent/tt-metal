// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// FlashFused reader: feeds the production sdpa_standard() compute. Reads this core's Q head (one
// seq-tile chunk, DHt tiles) once, then the single MQA KV head chunk-by-chunk (Sk_chunk_t*DHt tiles).
void kernel_main() {
    constexpr uint32_t NQH = get_compile_time_arg_val(0);
    constexpr uint32_t DHt = get_compile_time_arg_val(1);
    constexpr uint32_t Kt = get_compile_time_arg_val(2);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(4);
    constexpr auto q_args = TensorAccessorArgs<5>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t q_head = get_arg_val<uint32_t>(3);
    const uint32_t kv_head = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;

    const uint32_t q_tb = get_tile_size(cb_q_in);
    const uint32_t k_tb = get_tile_size(cb_k_in);
    const uint32_t v_tb = get_tile_size(cb_v_in);
    const auto q_acc = TensorAccessor(q_args, q_addr, q_tb);
    const auto k_acc = TensorAccessor(k_args, k_addr, k_tb);
    const auto v_acc = TensorAccessor(v_args, v_addr, v_tb);

    // Q: head q_head, single seq-tile chunk -> DHt tiles. page = q_head*DHt + d.
    cb_reserve_back(cb_q_in, DHt);
    uint32_t l1 = get_write_ptr(cb_q_in);
    for (uint32_t d = 0; d < DHt; ++d) {
        noc_async_read_tile(q_head * DHt + d, q_acc, l1);
        l1 += q_tb;
    }
    noc_async_read_barrier();
    cb_push_back(cb_q_in, DHt);

    // K/V: single MQA KV head, chunk by chunk. page = (kv_head*Kt + kt)*DHt + d.
    const uint32_t kv_base = kv_head * Kt;
    for (uint32_t c = 0; c < k_num_chunks; ++c) {
        const uint32_t kt0 = c * Sk_chunk_t;
        cb_reserve_back(cb_k_in, Sk_chunk_t * DHt);
        l1 = get_write_ptr(cb_k_in);
        for (uint32_t kt = 0; kt < Sk_chunk_t; ++kt) {
            for (uint32_t d = 0; d < DHt; ++d) {
                noc_async_read_tile((kv_base + kt0 + kt) * DHt + d, k_acc, l1);
                l1 += k_tb;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_k_in, Sk_chunk_t * DHt);

        cb_reserve_back(cb_v_in, Sk_chunk_t * DHt);
        l1 = get_write_ptr(cb_v_in);
        for (uint32_t kt = 0; kt < Sk_chunk_t; ++kt) {
            for (uint32_t d = 0; d < DHt; ++d) {
                noc_async_read_tile((kv_base + kt0 + kt) * DHt + d, v_acc, l1);
                l1 += v_tb;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_v_in, Sk_chunk_t * DHt);
    }
    (void)NQH;
}
