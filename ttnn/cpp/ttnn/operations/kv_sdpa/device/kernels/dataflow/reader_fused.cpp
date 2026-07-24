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
    constexpr bool use_provided_mask = get_compile_time_arg_val(7) == 1;
    constexpr uint32_t suffix_Kt = Kt - prefix_Kt;
    constexpr auto q_args = TensorAccessorArgs<8>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto pk_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto pv_args = TensorAccessorArgs<pk_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<pv_args.next_compile_time_args_offset()>();

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t q_head = get_arg_val<uint32_t>(3);
    const uint32_t kv_head = get_arg_val<uint32_t>(4);
    const uint32_t pk_addr = get_arg_val<uint32_t>(5);
    const uint32_t pv_addr = get_arg_val<uint32_t>(6);
    const uint32_t mask_addr = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;

    const uint32_t q_tb = get_tile_size(cb_q_in);
    const uint32_t k_tb = get_tile_size(cb_k_in);
    const uint32_t v_tb = get_tile_size(cb_v_in);
    const uint32_t mask_tb = get_tile_size(cb_mask_in);
    const auto q_acc = TensorAccessor(q_args, q_addr, q_tb);
    const auto k_acc = TensorAccessor(k_args, k_addr, k_tb);
    const auto v_acc = TensorAccessor(v_args, v_addr, v_tb);
    const auto pk_acc = TensorAccessor(pk_args, pk_addr, k_tb);
    const auto pv_acc = TensorAccessor(pv_args, pv_addr, v_tb);
    const auto mask_acc = TensorAccessor(mask_args, mask_addr, mask_tb);

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
        const uint32_t k_base = get_write_ptr(cb_k_in);
        cb_reserve_back(cb_v_in, Sk_chunk_t * DHt);
        uint32_t lv = get_write_ptr(cb_v_in);
        // K is written transposed into cb_k_in: tile (seq=kt, dim=d) -> offset (d*Sk_chunk_t + kt),
        // i.e. a [DHt x Sk_chunk_t] tile grid, as sdpa_standard's QK^T matmul expects (the production
        // SDPA reader likewise reads K with transpose=true). V stays seq-major [Sk_chunk_t x DHt].
        for (uint32_t kt = 0; kt < Sk_chunk_t; ++kt) {
            const uint32_t g = kt0 + kt;
            if (has_past && g < prefix_Kt) {
                const uint32_t base = (kv_head * prefix_Kt + g) * DHt;
                for (uint32_t d = 0; d < DHt; ++d) {
                    noc_async_read_tile(base + d, pk_acc, k_base + (d * Sk_chunk_t + kt) * k_tb);
                    noc_async_read_tile(base + d, pv_acc, lv + d * v_tb);
                }
            } else {
                const uint32_t base = (kv_head * suffix_Kt + (g - prefix_Kt)) * DHt;
                for (uint32_t d = 0; d < DHt; ++d) {
                    noc_async_read_tile(base + d, k_acc, k_base + (d * Sk_chunk_t + kt) * k_tb);
                    noc_async_read_tile(base + d, v_acc, lv + d * v_tb);
                }
            }
            lv += DHt * v_tb;
        }
        // Provided-mask path: read this chunk's mask tiles into cb_mask_in, aligned to the SAME
        // K-chunk iteration the K/V tiles use. Mask is [1, 1, Sq==1 tile, Kt]; with one Sq tile-row
        // the page index of KV column-tile g is exactly g (row-major over a 1 x Kt tile grid). The
        // folded KV ordering is [prefix(0..prefix_Kt) ; suffix(prefix_Kt..Kt)], i.e. the same global
        // index g the loop above uses, so the mask column ordering matches the folded KV by
        // construction (fold and non-fold both produce KV in [0..Kt) global order).
        if constexpr (use_provided_mask) {
            cb_reserve_back(cb_mask_in, Sk_chunk_t);
            uint32_t lm = get_write_ptr(cb_mask_in);
            for (uint32_t kt = 0; kt < Sk_chunk_t; ++kt) {
                noc_async_read_tile(kt0 + kt, mask_acc, lm);
                lm += mask_tb;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_k_in, Sk_chunk_t * DHt);
        cb_push_back(cb_v_in, Sk_chunk_t * DHt);
        if constexpr (use_provided_mask) {
            cb_push_back(cb_mask_in, Sk_chunk_t);
        }
    }
    (void)NQH;
}
