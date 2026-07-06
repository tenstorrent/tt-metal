// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Flash Attention — Reader Kernel (NCRISC)
// Streams Q, K, V tile blocks (and optional attn_mask) from DRAM into L1 CBs.
//
// CT arg layout (scalar first, then TensorAccessorArgs):
//   [0] B           — batch size
//   [1] H           — num heads
//   [2] S_q_t       — S_q in tiles
//   [3] S_kv_t      — S_kv in tiles
//   [4] D_t         — head dim in tiles
//   [5] num_q_blocks — number of Q blocks
//   [6] num_kv_blocks — number of KV blocks
//   [7] B_q          — Q block size in tiles
//   [8] B_kv         — KV block size in tiles
//   [9] has_mask     — 1 if attn_mask provided
//   [10] mask_per_head — 1 if mask has H heads, 0 if broadcast (1 head)
//   [11..] Q TensorAccessorArgs
//   [..] K TensorAccessorArgs
//   [..] V TensorAccessorArgs
//   [..] Mask TensorAccessorArgs (placeholder if no mask)
//
// RT arg layout:
//   [0] q_addr
//   [1] k_addr
//   [2] v_addr
//   [3] mask_addr
//   [4] b_idx  — batch index for this core
//   [5] h_idx  — head index for this core

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

// Scale tile fill — writes a uniform tile filled with a float value to a CB.
// Used for the attention scale factor (1/sqrt(D) or user-provided).
// The scale needs a uniform fill across the entire 32x32 tile (not row-0/col-0
// like reduce scalers), so we can't use prepare_reduce_scaler.
// This fills the tile in L1 directly using a packed bf16 pattern.
FORCE_INLINE void fill_scale_tile(uint32_t cb_id, uint32_t scale_bits_packed) {
    cb_reserve_back(cb_id, 1);
    uint32_t write_addr = get_write_ptr(cb_id);
    // Fill the tile with the packed bf16 value (each uint32 = 2 identical bf16 values)
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);
    for (uint32_t i = 0; i < 512; ++i) {  // 32x32 bf16 = 1024 elements = 512 uint32
        ptr[i] = scale_bits_packed;
    }
    cb_push_back(cb_id, 1);
}

void kernel_main() {
    // Scalar CT args
    const uint32_t B = get_compile_time_arg_val(0);
    const uint32_t H = get_compile_time_arg_val(1);
    const uint32_t S_q_t = get_compile_time_arg_val(2);
    const uint32_t S_kv_t = get_compile_time_arg_val(3);
    const uint32_t D_t = get_compile_time_arg_val(4);
    const uint32_t num_q_blocks = get_compile_time_arg_val(5);
    const uint32_t num_kv_blocks = get_compile_time_arg_val(6);
    constexpr uint32_t B_q = get_compile_time_arg_val(7);
    constexpr uint32_t B_kv = get_compile_time_arg_val(8);
    const uint32_t has_mask = get_compile_time_arg_val(9);
    const uint32_t mask_per_head = get_compile_time_arg_val(10);

    // TensorAccessorArgs start at CT offset 11
    constexpr auto q_args = TensorAccessorArgs<11>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();

    // RT args
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t mask_addr = get_arg_val<uint32_t>(3);
    const uint32_t b_idx = get_arg_val<uint32_t>(4);
    const uint32_t h_idx = get_arg_val<uint32_t>(5);

    // CB indices
    constexpr uint32_t cb_q = 0;
    constexpr uint32_t cb_k = 1;
    constexpr uint32_t cb_v = 2;
    constexpr uint32_t cb_attn_mask = 3;
    constexpr uint32_t cb_scale = 4;
    constexpr uint32_t cb_scaler = 31;

    const uint32_t tile_bytes = get_tile_size(cb_q);

    // Construct TensorAccessors
    const auto q_accessor = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k_accessor = TensorAccessor(k_args, k_addr, tile_bytes);
    const auto v_accessor = TensorAccessor(v_args, v_addr, tile_bytes);
    const auto mask_accessor = TensorAccessor(mask_args, mask_addr, tile_bytes);

    // Tensor layout: (B, H, S_t, D_t) — tiles are laid out as B*H*S_t*D_t pages
    // For Q: tile (b, h, sq, d) is at index b*H*S_q_t*D_t + h*S_q_t*D_t + sq*D_t + d
    const uint32_t q_tile_base = b_idx * H * S_q_t * D_t + h_idx * S_q_t * D_t;
    // For K/V: tile (b, h, skv, d) is at index b*H*S_kv_t*D_t + h*S_kv_t*D_t + skv*D_t + d
    const uint32_t kv_tile_base = b_idx * H * S_kv_t * D_t + h_idx * S_kv_t * D_t;

    // Mask tile base: (B, H_m, S_q, S_kv) — H_m is 1 or H
    uint32_t mask_tile_base = 0;
    if (has_mask) {
        uint32_t mask_H = mask_per_head ? H : 1;
        uint32_t mask_h_idx = mask_per_head ? h_idx : 0;
        // mask layout: (B, H_m, S_q_t, S_kv_t)
        mask_tile_base = b_idx * mask_H * S_q_t * S_kv_t + mask_h_idx * S_q_t * S_kv_t;
    }

    // Push reduce scalers per KV block (MAX + SUM)
    // The compute kernel calls reduce<MAX, REDUCE_ROW> and reduce<SUM, REDUCE_ROW>
    // Each needs a scaler tile. We push 2 scaler tiles per KV block.
    // MAX scaler: row-0 fill, value 1.0
    // SUM scaler: col-0 fill, value 1.0
    // calculate_and_prepare_reduce_scaler handles the fill pattern automatically.

    // Outer loop over Q blocks
    for (uint32_t qb = 0; qb < num_q_blocks; qb++) {
        // Push Q block for this Q-block iteration
        // Q block tiles: B_q rows × D_t cols, starting at tile row qb*B_q
        for (uint32_t d = 0; d < D_t; d++) {
            for (uint32_t sq = 0; sq < B_q; sq++) {
                uint32_t q_tile_row = qb * B_q + sq;
                if (q_tile_row >= S_q_t) {
                    break;
                }
                uint32_t tile_id = q_tile_base + q_tile_row * D_t + d;
                cb_reserve_back(cb_q, 1);
                uint32_t l1_write_addr = get_write_ptr(cb_q);
                noc_async_read_tile(tile_id, q_accessor, l1_write_addr);
                noc_async_read_barrier();
                cb_push_back(cb_q, 1);
            }
        }

        // Inner loop over KV blocks
        for (uint32_t kvb = 0; kvb < num_kv_blocks; kvb++) {
            // Push K block: B_kv rows × D_t cols, transposed for QK^T
            // K layout: (B, H, S_kv_t, D_t) — we read tiles row by row
            for (uint32_t d = 0; d < D_t; d++) {
                for (uint32_t skv = 0; skv < B_kv; skv++) {
                    uint32_t k_tile_row = kvb * B_kv + skv;
                    if (k_tile_row >= S_kv_t) {
                        break;
                    }
                    uint32_t tile_id = kv_tile_base + k_tile_row * D_t + d;
                    cb_reserve_back(cb_k, 1);
                    uint32_t l1_write_addr = get_write_ptr(cb_k);
                    noc_async_read_tile(tile_id, k_accessor, l1_write_addr);
                    noc_async_read_barrier();
                    cb_push_back(cb_k, 1);
                }
            }

            // Push V block: B_kv rows × D_t cols
            for (uint32_t d = 0; d < D_t; d++) {
                for (uint32_t skv = 0; skv < B_kv; skv++) {
                    uint32_t v_tile_row = kvb * B_kv + skv;
                    if (v_tile_row >= S_kv_t) {
                        break;
                    }
                    uint32_t tile_id = kv_tile_base + v_tile_row * D_t + d;
                    cb_reserve_back(cb_v, 1);
                    uint32_t l1_write_addr = get_write_ptr(cb_v);
                    noc_async_read_tile(tile_id, v_accessor, l1_write_addr);
                    noc_async_read_barrier();
                    cb_push_back(cb_v, 1);
                }
            }

            // Push mask block if present
            // Mask layout: (B, H_m, S_q_t, S_kv_t) — tiles laid out as B*H_m*S_q_t*S_kv_t
            if (has_mask) {
                for (uint32_t skv = 0; skv < B_kv; skv++) {
                    uint32_t m_skv = kvb * B_kv + skv;
                    if (m_skv >= S_kv_t) {
                        break;
                    }
                    for (uint32_t sq = 0; sq < B_q; sq++) {
                        uint32_t m_sq = qb * B_q + sq;
                        if (m_sq >= S_q_t) {
                            break;
                        }
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
