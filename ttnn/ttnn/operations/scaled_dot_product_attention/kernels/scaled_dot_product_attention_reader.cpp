// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for Flash-Attention SDPA.
//
// Per work item (b, h_q, qb):
//   - Q block  -> cb_q_in     : natural [B_q, DHt] layout (seq-major).
//   - per KV block j in [0, n_kv):
//       K block -> cb_k_in     : GRID-TRANSPOSED to [DHt, B_kv]. K is stored in
//                                DRAM as [S_kv, D] (seq x head); tile (s, d) is
//                                placed at cb position d*B_kv + s. Combined with
//                                the matmul's per-tile transpose=true this yields
//                                K^T = [DHt, B_kv] for the QK matmul.
//       V block -> cb_v_in     : natural [B_kv, vDHt].
//       mask    -> cb_mask_in  : natural [B_q, B_kv] (only when has_mask).
//
// GQA/MQA: KV head index h_kv = h_q / (H_q / H_kv). Mask head = 0 when the mask
// has a single head, else h_q.
//
// Two reduce scaler tiles (MAX row-0 fill, SUM col-0 fill) are written once at
// startup; the reduce helpers wait on them but never pop.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    uint32_t start_work = get_arg_val<uint32_t>(0);
    uint32_t num_work = get_arg_val<uint32_t>(1);
    uint32_t q_addr = get_arg_val<uint32_t>(2);
    uint32_t k_addr = get_arg_val<uint32_t>(3);
    uint32_t v_addr = get_arg_val<uint32_t>(4);
    uint32_t mask_addr = get_arg_val<uint32_t>(5);

    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t H_q = get_compile_time_arg_val(1);
    constexpr uint32_t H_kv = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_t = get_compile_time_arg_val(3);
    constexpr uint32_t Sk_t = get_compile_time_arg_val(4);
    constexpr uint32_t DHt = get_compile_time_arg_val(5);
    constexpr uint32_t vDHt = get_compile_time_arg_val(6);
    constexpr uint32_t B_q = get_compile_time_arg_val(7);
    constexpr uint32_t B_kv = get_compile_time_arg_val(8);
    constexpr uint32_t n_q = get_compile_time_arg_val(9);
    constexpr uint32_t n_kv = get_compile_time_arg_val(10);
    constexpr uint32_t has_mask = get_compile_time_arg_val(11);
    constexpr uint32_t mask_H = get_compile_time_arg_val(12);

    constexpr uint32_t cb_q_in = 0;
    constexpr uint32_t cb_k_in = 1;
    constexpr uint32_t cb_v_in = 2;
    constexpr uint32_t cb_mask_in = 3;
    constexpr uint32_t cb_max_scaler = 8;
    constexpr uint32_t cb_sum_scaler = 9;

    // Chained TensorAccessorArgs: scalar CT args occupy 0..12.
    constexpr auto q_args = TensorAccessorArgs<13>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto mask_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();

    const uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    const uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    const uint32_t v_tile_bytes = get_tile_size(cb_v_in);

    // Reduce scalers (value 1.0): MAX -> row-0 fill, SUM+REDUCE_ROW -> col-0
    // (matmul-path) fill. Written once; never popped.
    dataflow_kernel_lib::prepare_reduce_scaler<cb_max_scaler, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>(
        1.0f);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_sum_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        1.0f);

    const auto q_acc = TensorAccessor(q_args, q_addr, q_tile_bytes);
    const auto k_acc = TensorAccessor(k_args, k_addr, k_tile_bytes);
    const auto v_acc = TensorAccessor(v_args, v_addr, v_tile_bytes);

    constexpr uint32_t q_block_tiles = B_q * DHt;
    constexpr uint32_t k_block_tiles = B_kv * DHt;
    constexpr uint32_t v_block_tiles = B_kv * vDHt;
    constexpr uint32_t mask_block_tiles = B_q * B_kv;
    constexpr uint32_t kv_grouping = H_q / H_kv;

    for (uint32_t w = start_work; w < start_work + num_work; ++w) {
        // decode flat work index -> (b, h_q, qb)
        const uint32_t qb = w % n_q;
        const uint32_t r = w / n_q;
        const uint32_t h_q = r % H_q;
        const uint32_t b = r / H_q;
        const uint32_t h_kv = h_q / kv_grouping;
        const uint32_t mh = (mask_H == 1) ? 0 : h_q;

        // --- Q block: natural [B_q, DHt] ---
        const uint32_t q_head_base = (b * H_q + h_q) * Sq_t;
        cb_reserve_back(cb_q_in, q_block_tiles);
        {
            const uint32_t wptr = get_write_ptr(cb_q_in);
            for (uint32_t qr = 0; qr < B_q; ++qr) {
                const uint32_t row_page = (q_head_base + qb * B_q + qr) * DHt;
                for (uint32_t d = 0; d < DHt; ++d) {
                    noc_async_read_tile(row_page + d, q_acc, wptr + (qr * DHt + d) * q_tile_bytes);
                }
            }
            noc_async_read_barrier();
        }
        cb_push_back(cb_q_in, q_block_tiles);

        const uint32_t kv_head_base = (b * H_kv + h_kv) * Sk_t;
        const uint32_t mask_head_base = (b * mask_H + mh) * Sq_t;

        for (uint32_t j = 0; j < n_kv; ++j) {
            // --- K block: grid-transposed [DHt, B_kv] ---
            cb_reserve_back(cb_k_in, k_block_tiles);
            {
                const uint32_t wptr = get_write_ptr(cb_k_in);
                for (uint32_t s = 0; s < B_kv; ++s) {
                    const uint32_t row_page = (kv_head_base + j * B_kv + s) * DHt;
                    for (uint32_t d = 0; d < DHt; ++d) {
                        const uint32_t cb_pos = d * B_kv + s;  // [DHt, B_kv] layout
                        noc_async_read_tile(row_page + d, k_acc, wptr + cb_pos * k_tile_bytes);
                    }
                }
                noc_async_read_barrier();
            }
            cb_push_back(cb_k_in, k_block_tiles);

            // --- V block: natural [B_kv, vDHt] ---
            cb_reserve_back(cb_v_in, v_block_tiles);
            {
                const uint32_t wptr = get_write_ptr(cb_v_in);
                for (uint32_t s = 0; s < B_kv; ++s) {
                    const uint32_t row_page = (kv_head_base + j * B_kv + s) * vDHt;
                    for (uint32_t d = 0; d < vDHt; ++d) {
                        noc_async_read_tile(row_page + d, v_acc, wptr + (s * vDHt + d) * v_tile_bytes);
                    }
                }
                noc_async_read_barrier();
            }
            cb_push_back(cb_v_in, v_block_tiles);

            // --- mask block: natural [B_q, B_kv] ---
            if constexpr (has_mask) {
                const auto mask_acc = TensorAccessor(mask_args, mask_addr, get_tile_size(cb_mask_in));
                const uint32_t mask_tile_bytes = get_tile_size(cb_mask_in);
                cb_reserve_back(cb_mask_in, mask_block_tiles);
                const uint32_t wptr = get_write_ptr(cb_mask_in);
                for (uint32_t qr = 0; qr < B_q; ++qr) {
                    const uint32_t row_page = (mask_head_base + qb * B_q + qr) * Sk_t;
                    for (uint32_t s = 0; s < B_kv; ++s) {
                        const uint32_t page = row_page + j * B_kv + s;
                        noc_async_read_tile(page, mask_acc, wptr + (qr * B_kv + s) * mask_tile_bytes);
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_mask_in, mask_block_tiles);
            }
        }
    }
}
