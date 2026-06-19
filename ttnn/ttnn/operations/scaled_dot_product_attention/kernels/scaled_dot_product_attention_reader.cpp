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

// Generate a persistent additive KV-column pad-mask block into cb_pad_mask.
// The block is [B_q, B_kv] tiles (tile-row-major, matching cb_qk). Every element
// is 0 except columns [valid_cols, 32) of each row's LAST tile-column (within-
// block column B_kv-1), which are set to -inf. This is the partial-S_kv tile
// (global KV tile Sk_t-1) whose padded columns would otherwise contribute
// exp(0 - rowmax) to the softmax denominator. Generated ONCE at startup
// (work-item-invariant) and never popped; the compute adds it held on the last
// kv block. Tile element layout is face-major: a 32x32 tile is 4 16x16 faces
// ordered (top-left, top-right, bottom-left, bottom-right), each row-major, so
// element (row, col) lives at face*256 + (row%16)*16 + (col%16).
template <uint32_t cb, uint32_t Bq, uint32_t Bkv, uint32_t valid_cols, bool is_fp32>
inline void generate_kv_pad_mask() {
    cb_reserve_back(cb, Bq * Bkv);
    const uint32_t base = get_write_ptr(cb);
    const uint32_t tile_bytes = get_tile_size(cb);

    // 1) Zero the whole block (tile_bytes is a multiple of 4 for bf16 and fp32).
    volatile tt_l1_ptr uint32_t* zero_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    const uint32_t total_words = (Bq * Bkv * tile_bytes) >> 2;
    for (uint32_t i = 0; i < total_words; ++i) {
        zero_ptr[i] = 0;
    }

    // 2) Write -inf into the padded columns of each row's last-column tile.
    for (uint32_t r = 0; r < Bq; ++r) {
        const uint32_t tile_idx = r * Bkv + (Bkv - 1);
        const uint32_t tile_base = base + tile_idx * tile_bytes;
        for (uint32_t row = 0; row < 32; ++row) {
            for (uint32_t col = valid_cols; col < 32; ++col) {
                const uint32_t face = (row / 16) * 2 + (col / 16);
                const uint32_t in_face = (row % 16) * 16 + (col % 16);
                const uint32_t elem = face * 256 + in_face;
                if constexpr (is_fp32) {
                    volatile tt_l1_ptr uint32_t* e = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_base);
                    e[elem] = 0xFF800000u;  // -inf, fp32
                } else {
                    volatile tt_l1_ptr uint16_t* e = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_base);
                    e[elem] = 0xFF80u;  // -inf, bf16
                }
            }
        }
    }
    cb_push_back(cb, Bq * Bkv);
}

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
    constexpr uint32_t kv_partial = get_compile_time_arg_val(13);  // S_kv % 32 (0 => aligned)
    constexpr uint32_t pad_mask_is_fp32 = get_compile_time_arg_val(14);

    constexpr uint32_t cb_q_in = 0;
    constexpr uint32_t cb_k_in = 1;
    constexpr uint32_t cb_v_in = 2;
    constexpr uint32_t cb_mask_in = 3;
    constexpr uint32_t cb_pad_mask = 4;
    constexpr uint32_t cb_max_scaler = 8;
    constexpr uint32_t cb_sum_scaler = 9;

    // Chained TensorAccessorArgs: scalar CT args occupy 0..14.
    constexpr auto q_args = TensorAccessorArgs<15>();
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

    // KV-column pad mask (only when S_kv is non-tile-aligned). Generated once;
    // never popped. The compute adds it on the last kv block before the rowmax.
    if constexpr (kv_partial != 0) {
        generate_kv_pad_mask<cb_pad_mask, B_q, B_kv, kv_partial, pad_mask_is_fp32 != 0>();
    }

    const auto q_acc = TensorAccessor(q_args, q_addr, q_tile_bytes);
    const auto k_acc = TensorAccessor(k_args, k_addr, k_tile_bytes);
    const auto v_acc = TensorAccessor(v_args, v_addr, v_tile_bytes);
    // Mask accessor + tile size are loop-invariant — build once, not per
    // KV-block (a TensorAccessor and get_tile_size both recompute otherwise).
    const uint32_t mask_tile_bytes = has_mask ? get_tile_size(cb_mask_in) : 0;
    const auto mask_acc = TensorAccessor(mask_args, mask_addr, mask_tile_bytes);

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
