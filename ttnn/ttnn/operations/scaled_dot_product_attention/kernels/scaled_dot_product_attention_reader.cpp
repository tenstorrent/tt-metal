// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Flash-Attention reader kernel.
//
// Prepares the two reduce scaler tiles (resident for the whole kernel), then
// for each assigned work-unit (b, h, q-chunk) streams the Q-chunk once and
// every KV-chunk (K, V, and optionally the mask tile) into the input CBs.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

// Generate the diagonal triangular causal-bias tile directly into L1 (bf16).
// Additive form: element (row, col) -> 0 where col <= row (attend), neg_bits
// (a large negative) where col > row (mask). Tile layout is 4 faces of 16x16
// (row-major within face; faces ordered TL, TR, BL, BR).
inline void gen_causal_mask_tile(uint32_t cb_id, uint16_t neg_bits) {
    constexpr uint32_t FACE = 16;
    cb_reserve_back(cb_id, 1);
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id));
    for (uint32_t face = 0; face < 4; ++face) {
        const uint32_t row_off = (face >= 2) ? FACE : 0;
        const uint32_t col_off = (face & 1U) ? FACE : 0;
        for (uint32_t h = 0; h < FACE; ++h) {
            const uint32_t row = row_off + h;
            for (uint32_t w = 0; w < FACE; ++w) {
                const uint32_t col = col_off + w;
                *ptr++ = (col <= row) ? uint16_t(0) : neg_bits;
            }
        }
    }
    cb_push_back(cb_id, 1);
}

// R3 — generate the KV-sequence column edge-mask tile (bf16) into L1.
// Additive form: element (row, col) -> 0 where col < valid_cols (a real key
// position), neg_bits (a large negative) where col >= valid_cols (a padding
// key column that must not pollute the softmax). Broadcast across all rows.
inline void gen_edge_mask_tile(uint32_t cb_id, uint32_t valid_cols, uint16_t neg_bits) {
    constexpr uint32_t FACE = 16;
    cb_reserve_back(cb_id, 1);
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id));
    for (uint32_t face = 0; face < 4; ++face) {
        const uint32_t col_off = (face & 1U) ? FACE : 0;
        for (uint32_t h = 0; h < FACE; ++h) {
            for (uint32_t w = 0; w < FACE; ++w) {
                const uint32_t col = col_off + w;
                *ptr++ = (col < valid_cols) ? uint16_t(0) : neg_bits;
            }
        }
    }
    cb_push_back(cb_id, 1);
}

void kernel_main() {
    constexpr uint32_t H_q = get_compile_time_arg_val(0);
    constexpr uint32_t H_kv = get_compile_time_arg_val(1);
    constexpr uint32_t Sq_t = get_compile_time_arg_val(2);
    constexpr uint32_t Skv_t = get_compile_time_arg_val(3);
    constexpr uint32_t d_t = get_compile_time_arg_val(4);
    constexpr uint32_t group = get_compile_time_arg_val(5);  // H_q / H_kv
    constexpr uint32_t has_mask = get_compile_time_arg_val(6);
    constexpr uint32_t mask_H = get_compile_time_arg_val(7);  // mask num-heads (1 or H_q)
    constexpr uint32_t mask_B = get_compile_time_arg_val(8);  // mask batch (1 or B)
    constexpr uint32_t causal = get_compile_time_arg_val(9);  // native causal masking
    constexpr uint16_t causal_neg_bits = get_compile_time_arg_val(10);  // bf16 −inf bias bits
    constexpr uint32_t kv_edge = get_compile_time_arg_val(11);          // R3 — KV-seq edge masking active
    constexpr uint32_t kv_valid_last = get_compile_time_arg_val(12);    // valid key cols in last KV tile
    constexpr uint32_t Bkv_t = get_compile_time_arg_val(13);            // R5 — KV tiles per chunk
    // Non-causal KV-chunk count. Bkv_t divides Skv_t when Bkv_t > 1
    // (host-enforced); causal/edge keep Bkv_t == 1 so Nkv == Skv_t and j is a
    // tile index (diagonal/edge logic unchanged).
    constexpr uint32_t Nkv = Skv_t / Bkv_t;

    constexpr auto q_args = TensorAccessorArgs<14>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto mask_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();

    const uint32_t start_unit = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);
    const uint32_t q_addr = get_arg_val<uint32_t>(2);
    const uint32_t k_addr = get_arg_val<uint32_t>(3);
    const uint32_t v_addr = get_arg_val<uint32_t>(4);
    const uint32_t mask_addr = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_q_in = 0;
    constexpr uint32_t cb_k_in = 1;
    constexpr uint32_t cb_v_in = 2;
    constexpr uint32_t cb_mask_in = 3;
    constexpr uint32_t cb_edge_mask = 4;  // R3 — KV-seq column edge mask
    constexpr uint32_t cb_scaler_max = 8;
    constexpr uint32_t cb_scaler_sum = 9;

    // Reduce scalers (value 1.0, pool-type-aware fill), resident for the kernel.
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler_max, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>(
        1.0f);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler_sum, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        1.0f);

    const uint32_t tile_bytes = get_tile_size(cb_q_in);
    const auto q_acc = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k_acc = TensorAccessor(k_args, k_addr, tile_bytes);
    const auto v_acc = TensorAccessor(v_args, v_addr, tile_bytes);
    const auto mask_acc = TensorAccessor(mask_args, mask_addr, tile_bytes);

    for (uint32_t u = start_unit; u < start_unit + num_units; ++u) {
        // Decode unit -> (b, h, qc).
        const uint32_t qc = u % Sq_t;
        const uint32_t tmp = u / Sq_t;
        const uint32_t h = tmp % H_q;
        const uint32_t b = tmp / H_q;
        const uint32_t h_kv = h / group;
        // Mask broadcasting: dim0 collapses to 0 when mask_B==1 (batch-broadcast);
        // dim1 collapses to 0 when mask_H==1 (head-broadcast).
        const uint32_t mask_h = (mask_H == 1) ? 0 : h;
        const uint32_t mask_b = (mask_B == 1) ? 0 : b;

        // --- Q-chunk: d_t head tiles of query tile-row qc ---
        const uint32_t q_base = ((b * H_q + h) * Sq_t + qc) * d_t;
        cb_reserve_back(cb_q_in, d_t);
        uint32_t q_wr = get_write_ptr(cb_q_in);
        for (uint32_t dd = 0; dd < d_t; ++dd) {
            noc_async_read_page(q_base + dd, q_acc, q_wr);
            q_wr += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_q_in, d_t);

        // Causal: query-chunk qc attends only to key-chunks j <= qc (the
        // diagonal block j==qc is partially masked; j>qc is fully future ->
        // skipped entirely). Requires S_q==S_kv (causal+cross is excluded).
        // R5: non-causal streams Nkv chunks of Bkv_t tiles each; causal keeps
        // Bkv_t == 1 so Nkv == Skv_t and j is a tile index.
        const uint32_t kv_count = causal ? (qc + 1) : Nkv;

        // --- KV loop ---
        for (uint32_t j = 0; j < kv_count; ++j) {
            // First key tile-row of this chunk; the chunk spans key tile-rows
            // [kv0, kv0 + Bkv_t).
            const uint32_t kv0 = j * Bkv_t;
            const uint32_t kv_tile_base = (b * H_kv + h_kv) * Skv_t;

            // K block — laid out K-major for the QK^T matmul in1 ([d_t][Bkv_t]):
            //   dd outer, key inner -> tile index dd*Bkv_t + key. (Bkv_t == 1
            //   recovers the original dd-only order.)
            cb_reserve_back(cb_k_in, Bkv_t * d_t);
            uint32_t k_wr = get_write_ptr(cb_k_in);
            for (uint32_t dd = 0; dd < d_t; ++dd) {
                for (uint32_t kk = 0; kk < Bkv_t; ++kk) {
                    noc_async_read_page((kv_tile_base + (kv0 + kk)) * d_t + dd, k_acc, k_wr);
                    k_wr += tile_bytes;
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_k_in, Bkv_t * d_t);

            // V block — laid out K-major for the PV matmul in1 ([Bkv_t][d_t]):
            //   key outer, dd inner -> tile index key*d_t + dd (natural DRAM
            //   order). (Bkv_t == 1 recovers the original single tile-row.)
            cb_reserve_back(cb_v_in, Bkv_t * d_t);
            uint32_t v_wr = get_write_ptr(cb_v_in);
            for (uint32_t kk = 0; kk < Bkv_t; ++kk) {
                for (uint32_t dd = 0; dd < d_t; ++dd) {
                    noc_async_read_page((kv_tile_base + (kv0 + kk)) * d_t + dd, v_acc, v_wr);
                    v_wr += tile_bytes;
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_v_in, Bkv_t * d_t);

            if constexpr (has_mask) {
                // Custom mask block: Bkv_t tiles for (qc, key) over the chunk.
                const uint32_t mask_row_base = ((mask_b * mask_H + mask_h) * Sq_t + qc) * Skv_t;
                cb_reserve_back(cb_mask_in, Bkv_t);
                uint32_t m_wr = get_write_ptr(cb_mask_in);
                for (uint32_t kk = 0; kk < Bkv_t; ++kk) {
                    noc_async_read_page(mask_row_base + kv0 + kk, mask_acc, m_wr);
                    m_wr += tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb_mask_in, Bkv_t);
            } else if constexpr (causal) {
                // Only the diagonal-straddling block (j == qc) needs the
                // triangular bias; j < qc is fully past (unmasked).
                if (j == qc) {
                    gen_causal_mask_tile(cb_mask_in, causal_neg_bits);
                }
            }

            // R3 — KV-sequence edge mask. Independent of cb_mask_in: pushed on
            // the last KV chunk only (j == Skv_t-1), for the {none, custom}
            // modes (kv_edge is 0 under causal). Forces the padding key columns
            // of the score tile to −inf so they drop out of the softmax.
            if constexpr (kv_edge) {
                if (j == Skv_t - 1) {
                    gen_edge_mask_tile(cb_edge_mask, kv_valid_last, causal_neg_bits);
                }
            }
        }
    }
}
