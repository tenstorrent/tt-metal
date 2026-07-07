// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Flash Attention — Reader Kernel (NCRISC)
// Streams Q, K, V tile blocks (and optional attn_mask) from DRAM into L1 CBs.
//
// GQA/MQA support: K and V tensors have H_kv heads (H_kv <= H_q). Each Q head
// h_q maps to K/V head h_kv = h_q * H_kv / H_q (matching repeat_interleave
// broadcasting: each KV head is replicated H_q/H_kv times consecutively).
//
// Causal masking (is_causal=1): generates triangular mask on-device in L1.
// Three regions per (Q-block, KV-block):
//   - Past blocks (kvb < qb): push a zero mask (no masking needed)
//   - Future blocks (kvb > qb): push a -inf mask (entire block masked)
//   - Diagonal block (kvb == qb): generate per-element triangular mask
// The compute kernel still runs QK^T for all blocks, but the -inf mask makes
// future-block scores all -inf, which the online softmax naturally drops.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

// Tile face layout constants (32×32 tile = 4 faces of 16×16)
// Face 0: top-left,  Face 1: top-right
// Face 2: bottom-left, Face 3: bottom-right
constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t FACE_DIM = 16;

// IEEE 754 bit representations for compile-time template parameters
constexpr uint32_t FP32_ZERO_BITS = 0x00000000;     // 0.0f
constexpr uint32_t FP32_NEG_INF_BITS = 0xCE6E6B28;  // bit_cast<u32>(-1e9F)
constexpr uint16_t BF16_ZERO_BITS = 0x0000;         // 0.0 in bfloat16
constexpr uint16_t BF16_NEG_INF_BITS = 0xCE6E;      // upper 16 bits of -1e9F

// Generate a causal (lower triangular) mask tile directly in L1 CB memory.
// mask[row, col] = 0.0 (attend) if col <= row, else -1e9 (mask out).
// The tile is 32×32 with 4 faces of 16×16, laid out as:
//   Face 0 (rows 0-15, cols 0-15): lower-left triangle of top-left quadrant
//   Face 1 (rows 0-15, cols 16-31): all masked (cols > rows)
//   Face 2 (rows 16-31, cols 0-15): all attend (cols <= rows)
//   Face 3 (rows 16-31, cols 16-31): lower triangle of bottom-right quadrant
template <typename T, T zero_value, T neg_inf_value>
inline void fill_causal_mask_tile(uint32_t cb_id) {
    T* tile_ptr = reinterpret_cast<T*>(get_write_ptr(cb_id));
    for (uint32_t face = 0; face < 4; ++face) {
        const uint32_t face_row_offset = (face >= 2) ? FACE_DIM : 0;
        const uint32_t face_col_offset = (face & 1U) ? FACE_DIM : 0;
        for (uint32_t h = 0; h < FACE_DIM; ++h) {
            const uint32_t row = face_row_offset + h;
            for (uint32_t w = 0; w < FACE_DIM; ++w) {
                const uint32_t col = face_col_offset + w;
                *tile_ptr++ = (col <= row) ? zero_value : neg_inf_value;
            }
        }
    }
}

// Fill an entire tile with a constant value (for all-zero or all-neg-inf masks)
template <typename T, T value>
inline void fill_constant_tile(uint32_t cb_id) {
    T* tile_ptr = reinterpret_cast<T*>(get_write_ptr(cb_id));
    for (uint32_t i = 0; i < TILE_DIM * TILE_DIM; ++i) {
        *tile_ptr++ = value;
    }
}

void kernel_main() {
    // Scalar CT args
    const uint32_t B = get_compile_time_arg_val(0);
    const uint32_t H = get_compile_time_arg_val(1);  // H_q (Q num heads)
    const uint32_t S_q_t = get_compile_time_arg_val(2);
    const uint32_t S_kv_t = get_compile_time_arg_val(3);
    const uint32_t D_t = get_compile_time_arg_val(4);
    const uint32_t num_q_blocks = get_compile_time_arg_val(5);
    const uint32_t num_kv_blocks = get_compile_time_arg_val(6);
    constexpr uint32_t B_q = get_compile_time_arg_val(7);
    constexpr uint32_t B_kv = get_compile_time_arg_val(8);
    const uint32_t has_mask = get_compile_time_arg_val(9);
    const uint32_t mask_per_head = get_compile_time_arg_val(10);
    const uint32_t H_kv = get_compile_time_arg_val(11);  // K/V num heads (GQA/MQA)
    constexpr uint32_t is_causal = get_compile_time_arg_val(12);

    // TensorAccessorArgs start at CT offset 13
    constexpr auto q_args = TensorAccessorArgs<13>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();

    // RT args: [num_work_units, q_addr, k_addr, v_addr, mask_addr,
    //           b0, h0, b1, h1, ...]
    const uint32_t num_work_units = get_arg_val<uint32_t>(0);
    const uint32_t q_addr = get_arg_val<uint32_t>(1);
    const uint32_t k_addr = get_arg_val<uint32_t>(2);
    const uint32_t v_addr = get_arg_val<uint32_t>(3);
    const uint32_t mask_addr = get_arg_val<uint32_t>(4);

    // CB indices
    constexpr uint32_t cb_q = 0;
    constexpr uint32_t cb_k = 1;
    constexpr uint32_t cb_v = 2;
    constexpr uint32_t cb_attn_mask = 3;
    constexpr uint32_t cb_scaler = 31;

    const uint32_t tile_bytes = get_tile_size(cb_q);

    const auto q_accessor = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k_accessor = TensorAccessor(k_args, k_addr, tile_bytes);
    const auto v_accessor = TensorAccessor(v_args, v_addr, tile_bytes);
    const auto mask_accessor = TensorAccessor(mask_args, mask_addr, tile_bytes);

    // Determine mask tile data format for causal mask generation
    const uint32_t mask_tile_bytes = get_tile_size(cb_attn_mask);
    const bool mask_is_fp32 = (mask_tile_bytes >= 4 * TILE_DIM * TILE_DIM);

    // Loop over work units (multiple (B,H) pairs per core when B*H > num_cores)
    uint32_t rt_arg_idx = 5;  // b/h pairs start at RT arg index 5
    for (uint32_t wu = 0; wu < num_work_units; wu++) {
        const uint32_t b_idx = get_arg_val<uint32_t>(rt_arg_idx++);
        const uint32_t h_idx = get_arg_val<uint32_t>(rt_arg_idx++);

        // Tensor layout: (B, H, S_t, D_t) — tiles are laid out as B*H*S_t*D_t pages
        // Q has H (=H_q) heads; K/V have H_kv heads. For GQA/MQA (H_kv < H_q),
        // Q head h_idx maps to K/V head h_kv = h_idx * H_kv / H_q (repeat_interleave).
        const uint32_t q_tile_base = b_idx * H * S_q_t * D_t + h_idx * S_q_t * D_t;
        const uint32_t h_kv = (H_kv == H) ? h_idx : (h_idx * H_kv / H);
        const uint32_t kv_tile_base = b_idx * H_kv * S_kv_t * D_t + h_kv * S_kv_t * D_t;

        uint32_t mask_tile_base = 0;
        if (has_mask) {
            uint32_t mask_H = mask_per_head ? H : 1;
            uint32_t mask_h_idx = mask_per_head ? h_idx : 0;
            mask_tile_base = b_idx * mask_H * S_q_t * S_kv_t + mask_h_idx * S_q_t * S_kv_t;
        }

        // Outer loop over Q blocks
        for (uint32_t qb = 0; qb < num_q_blocks; qb++) {
            // Inner loop over KV blocks
            for (uint32_t kvb = 0; kvb < num_kv_blocks; kvb++) {
                // Causal mask: determine block relationship to diagonal
                // For causal (self-attention, S_q == S_kv), tile row/col indices
                // are qb*B_q + sq and kvb*B_kv + skv. With B_q=B_kv=1, the
                // diagonal is at qb == kvb.
                if constexpr (is_causal) {
                    if (kvb > qb) {
                        // Future block: entire tile is -inf.
                        // Still push Q/K/V (compute will run QK^T but scores
                        // become -inf after mask add → softmax drops them).
                        // But for efficiency, we can skip Q/K/V entirely and
                        // let compute skip the block. However, compute expects
                        // K/V tiles for the matmul. So we push dummy K/V tiles.
                        // Actually — we need to push SOMETHING for K/V or the
                        // matmul will hang waiting for input. The simplest approach:
                        // push the real Q/K/V tiles (they're just data), and
                        // push a -inf mask. The compute kernel adds the mask
                        // to scores, making them all -inf, which the online
                        // softmax naturally drops (exp(-inf) = 0).
                        // This is simpler than skipping and avoids CB sync issues.
                    }
                }

                // Push Q block: B_q rows × D_t cols, TileRowMajor order
                // Re-pushed per KV block since matmul pops Q tiles
                for (uint32_t sq = 0; sq < B_q; sq++) {
                    uint32_t q_tile_row = qb * B_q + sq;
                    for (uint32_t d = 0; d < D_t; d++) {
                        uint32_t tile_id = q_tile_base + q_tile_row * D_t + d;
                        cb_reserve_back(cb_q, 1);
                        uint32_t l1_write_addr = get_write_ptr(cb_q);
                        noc_async_read_tile(tile_id, q_accessor, l1_write_addr);
                        noc_async_read_barrier();
                        cb_push_back(cb_q, 1);
                    }
                }

                // Push K block: B_kv rows × D_t cols
                for (uint32_t skv = 0; skv < B_kv; skv++) {
                    uint32_t k_tile_row = kvb * B_kv + skv;
                    for (uint32_t d = 0; d < D_t; d++) {
                        uint32_t tile_id = kv_tile_base + k_tile_row * D_t + d;
                        cb_reserve_back(cb_k, 1);
                        uint32_t l1_write_addr = get_write_ptr(cb_k);
                        noc_async_read_tile(tile_id, k_accessor, l1_write_addr);
                        noc_async_read_barrier();
                        cb_push_back(cb_k, 1);
                    }
                }

                // Push V block: B_kv rows × D_t cols
                for (uint32_t skv = 0; skv < B_kv; skv++) {
                    uint32_t v_tile_row = kvb * B_kv + skv;
                    for (uint32_t d = 0; d < D_t; d++) {
                        uint32_t tile_id = kv_tile_base + v_tile_row * D_t + d;
                        cb_reserve_back(cb_v, 1);
                        uint32_t l1_write_addr = get_write_ptr(cb_v);
                        noc_async_read_tile(tile_id, v_accessor, l1_write_addr);
                        noc_async_read_barrier();
                        cb_push_back(cb_v, 1);
                    }
                }

                // Push mask block
                if constexpr (is_causal) {
                    // Causal mask: generate on-device based on block position
                    for (uint32_t sq = 0; sq < B_q; sq++) {
                        for (uint32_t skv = 0; skv < B_kv; skv++) {
                            cb_reserve_back(cb_attn_mask, 1);
                            uint32_t q_tile_idx = qb * B_q + sq;
                            uint32_t kv_tile_idx = kvb * B_kv + skv;
                            if (kv_tile_idx > q_tile_idx) {
                                // Future block: all -inf
                                if (mask_is_fp32) {
                                    fill_constant_tile<uint32_t, FP32_NEG_INF_BITS>(cb_attn_mask);
                                } else {
                                    fill_constant_tile<uint16_t, BF16_NEG_INF_BITS>(cb_attn_mask);
                                }
                            } else if (kv_tile_idx == q_tile_idx) {
                                // Diagonal block: triangular mask
                                if (mask_is_fp32) {
                                    fill_causal_mask_tile<uint32_t, FP32_ZERO_BITS, FP32_NEG_INF_BITS>(cb_attn_mask);
                                } else {
                                    fill_causal_mask_tile<uint16_t, BF16_ZERO_BITS, BF16_NEG_INF_BITS>(cb_attn_mask);
                                }
                            } else {
                                // Past block: all zeros (no masking)
                                if (mask_is_fp32) {
                                    fill_constant_tile<uint32_t, FP32_ZERO_BITS>(cb_attn_mask);
                                } else {
                                    fill_constant_tile<uint16_t, BF16_ZERO_BITS>(cb_attn_mask);
                                }
                            }
                            cb_push_back(cb_attn_mask, 1);
                        }
                    }
                } else if (has_mask) {
                    // Custom mask: stream from DRAM
                    for (uint32_t sq = 0; sq < B_q; sq++) {
                        uint32_t m_sq = qb * B_q + sq;
                        for (uint32_t skv = 0; skv < B_kv; skv++) {
                            uint32_t m_skv = kvb * B_kv + skv;
                            uint32_t tile_id = mask_tile_base + m_sq * S_kv_t + m_skv;
                            cb_reserve_back(cb_attn_mask, 1);
                            uint32_t l1_write_addr = get_write_ptr(cb_attn_mask);
                            noc_async_read_tile(tile_id, mask_accessor, l1_write_addr);
                            noc_async_read_barrier();
                            cb_push_back(cb_attn_mask, 1);
                        }
                    }
                }

                // Push reduce scalers (2 tiles: MAX then SUM)
                dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
                    cb_scaler,
                    ckernel::PoolType::MAX,
                    ckernel::ReduceDim::REDUCE_ROW>();
                dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
                    cb_scaler,
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_ROW>();
            }
        }
    }
}
