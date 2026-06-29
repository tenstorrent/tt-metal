// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Flash Attention reader kernel.
// Reads Q, K, V tiles from DRAM into L1 CBs.
// Prepares reduce scaler tiles and scale factor tile.
//
// When is_causal=True: generates causal (lower-triangular) mask tiles
// on-device per (Q-block, KV-block) pair instead of reading an additive
// mask tensor from DRAM. Three regions:
//   - fully-past (kv_end <= q_start): all-zero mask (attend to all)
//   - fully-future (kv_start >= q_end): all-(-inf) mask (attend to none)
//   - diagonal-straddling: per-element triangular mask (0 for col<=row,
//     -inf for col>row)
// The causal mask is always bf16 regardless of input dtype — the compute
// kernel's add helper handles format reconfig.
//
// Work distribution: each core processes its assigned (B, H) work units
// sequentially. For GQA/MQA, maps Q-head index to KV-head index.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

// CB indices
constexpr uint32_t cb_q = tt::CBIndex::c_0;
constexpr uint32_t cb_k = tt::CBIndex::c_1;
constexpr uint32_t cb_v = tt::CBIndex::c_2;
constexpr uint32_t cb_mask = tt::CBIndex::c_3;
constexpr uint32_t cb_scaler_max = 6;
constexpr uint32_t cb_scaler_sum = 7;
constexpr uint32_t cb_scale_factor = 5;

// Tile face layout constants
constexpr uint32_t FACE_SIZE = 16;
constexpr uint32_t FACE_ELEMS = FACE_SIZE * FACE_SIZE;  // 256
constexpr uint32_t TILE_ELEMS = 4 * FACE_ELEMS;         // 1024

// bf16 bit patterns
constexpr uint16_t BF16_ZERO = 0x0000;     // 0.0
constexpr uint16_t BF16_NEG_INF = 0xFF80;  // -inf

// Convert fp32 bits to bf16 bits (round-to-nearest-even)
inline uint16_t fp32_bits_to_bf16_bits(uint32_t fp32_bits) {
    uint16_t lsw = static_cast<uint16_t>(fp32_bits & 0xFFFF);
    uint16_t bias = 0x7FFFu + (lsw >> 15);
    uint32_t rounded = fp32_bits + bias;
    return static_cast<uint16_t>(rounded >> 16);
}

// Fill a tile with a uniform bf16 value (fast path for fully-past/fully-future)
inline void fill_mask_tile_uniform(volatile tt_l1_ptr uint16_t* tile, uint16_t val) {
    for (uint32_t i = 0; i < TILE_ELEMS; ++i) {
        tile[i] = val;
    }
}

// Generate a causal mask tile for a (qr, kc) tile pair within the current block.
// q_tile_global and kv_tile_global are the absolute tile indices.
// Causal condition: col > row → -inf, col <= row → 0.
// Uses face layout: face 0 (r0-15,c0-15), face 1 (r0-15,c16-31),
//                   face 2 (r16-31,c0-15), face 3 (r16-31,c16-31).
inline void generate_causal_mask_tile(
    volatile tt_l1_ptr uint16_t* tile, uint32_t q_tile_global, uint32_t kv_tile_global) {
    for (uint32_t fr = 0; fr < 2; ++fr) {
        for (uint32_t fc = 0; fc < 2; ++fc) {
            uint32_t face_base = (fr * 2 + fc) * FACE_ELEMS;
            for (uint32_t r = 0; r < FACE_SIZE; ++r) {
                uint32_t global_row = q_tile_global * 32 + fr * FACE_SIZE + r;
                for (uint32_t c = 0; c < FACE_SIZE; ++c) {
                    uint32_t global_col = kv_tile_global * 32 + fc * FACE_SIZE + c;
                    tile[face_base + r * FACE_SIZE + c] = (global_col > global_row) ? BF16_NEG_INF : BF16_ZERO;
                }
            }
        }
    }
}

void kernel_main() {
    // CT args: [has_mask, is_causal, H_q, H_kv, mask_is_per_head,
    //           ...Q_accessor, ...K_accessor, ...V_accessor, ...mask_accessor]
    constexpr uint32_t has_mask = get_compile_time_arg_val(0);
    constexpr uint32_t is_causal = get_compile_time_arg_val(1);
    constexpr uint32_t H_q = get_compile_time_arg_val(2);
    constexpr uint32_t H_kv = get_compile_time_arg_val(3);
    constexpr uint32_t mask_is_per_head = get_compile_time_arg_val(4);

    // Prepare reduce scalers (1.0 for both MAX and SUM)
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler
        <cb_scaler_max, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler
        <cb_scaler_sum, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    // RT args: [num_work_units, B_q_t, B_kv_t, D_t, S_q_tiles, S_kv_tiles, (b,h)*num_work_units, q_addr, k_addr,
    // v_addr, scale_bits, mask_addr]
    uint32_t rt_idx = 0;
    uint32_t num_work_units = get_arg_val<uint32_t>(rt_idx++);
    uint32_t B_q_t = get_arg_val<uint32_t>(rt_idx++);
    uint32_t B_kv_t = get_arg_val<uint32_t>(rt_idx++);
    uint32_t D_t = get_arg_val<uint32_t>(rt_idx++);
    uint32_t S_q_tiles = get_arg_val<uint32_t>(rt_idx++);
    uint32_t S_kv_tiles = get_arg_val<uint32_t>(rt_idx++);

    // Read work unit (b, h) pairs
    uint32_t work_b[16], work_h[16];
    for (uint32_t i = 0; i < num_work_units; ++i) {
        work_b[i] = get_arg_val<uint32_t>(rt_idx++);
        work_h[i] = get_arg_val<uint32_t>(rt_idx++);
    }

    uint32_t q_addr = get_arg_val<uint32_t>(rt_idx++);
    uint32_t k_addr = get_arg_val<uint32_t>(rt_idx++);
    uint32_t v_addr = get_arg_val<uint32_t>(rt_idx++);
    uint32_t scale_bits = get_arg_val<uint32_t>(rt_idx++);
    [[maybe_unused]] uint32_t mask_addr = get_arg_val<uint32_t>(rt_idx++);

    // Fill scale factor CB (1 tile with the scale value)
    cb_reserve_back(cb_scale_factor, 1);
    {
        uint16_t bf16_bits = fp32_bits_to_bf16_bits(scale_bits);
        auto ptr = reinterpret_cast<volatile uint16_t*>(get_write_ptr(cb_scale_factor));
        for (uint32_t i = 0; i < 1024; ++i) ptr[i] = bf16_bits;
    }
    cb_push_back(cb_scale_factor, 1);

    // TensorAccessor declarations — unconditional, chained offsets
    // CT args: [has_mask, is_causal, H_q, H_kv, mask_is_per_head, ...accessors]
    constexpr auto q_args = TensorAccessorArgs<5>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();

    const auto q_accessor = TensorAccessor(q_args, q_addr);
    const auto k_accessor = TensorAccessor(k_args, k_addr);
    const auto v_accessor = TensorAccessor(v_args, v_addr);
    [[maybe_unused]] const auto mask_accessor = TensorAccessor(mask_args, mask_addr);

    uint32_t h_q_div_h_kv = H_q / H_kv;
    uint32_t num_q_blocks = (S_q_tiles + B_q_t - 1) / B_q_t;
    uint32_t num_kv_blocks = (S_kv_tiles + B_kv_t - 1) / B_kv_t;

    for (uint32_t wu = 0; wu < num_work_units; ++wu) {
        uint32_t b = work_b[wu];
        uint32_t h_q = work_h[wu];
        uint32_t h_kv = h_q / h_q_div_h_kv;

        // Base tile indices for this (B, H) pair
        uint32_t q_base = b * H_q * S_q_tiles * D_t + h_q * S_q_tiles * D_t;
        uint32_t k_base = b * H_kv * S_kv_tiles * D_t + h_kv * S_kv_tiles * D_t;
        uint32_t v_base = k_base;  // V has same shape as K

        // Mask base: (B, H_mask, S_q_tiles, S_kv_tiles)
        uint32_t mask_h = mask_is_per_head ? h_q : 0;
        uint32_t mask_base = b * (mask_is_per_head ? H_q : 1) * S_q_tiles * S_kv_tiles
                           + mask_h * S_q_tiles * S_kv_tiles;

        for (uint32_t qb = 0; qb < num_q_blocks; ++qb) {
            uint32_t q_row_start = qb * B_q_t;

            // Push Q-block tiles: B_q_t rows × D_t cols (once, retained across KV-blocks by matmul)
            for (uint32_t r = 0; r < B_q_t; ++r) {
                for (uint32_t d = 0; d < D_t; ++d) {
                    cb_reserve_back(cb_q, 1);
                    noc_async_read_tile(q_base + (q_row_start + r) * D_t + d, q_accessor, get_write_ptr(cb_q));
                    noc_async_read_barrier();
                    cb_push_back(cb_q, 1);
                }
            }

            // Stream KV-blocks for this Q-block
            for (uint32_t kvb = 0; kvb < num_kv_blocks; ++kvb) {
                uint32_t kv_col_start = kvb * B_kv_t;

                // K tiles: B_kv_t cols × D_t rows (K is (S_kv, D) in tile layout, read as D_t rows × B_kv_t cols)
                for (uint32_t k_row = 0; k_row < D_t; ++k_row) {
                    for (uint32_t k_col = 0; k_col < B_kv_t; ++k_col) {
                        cb_reserve_back(cb_k, 1);
                        noc_async_read_tile(
                            k_base + (kv_col_start + k_col) * D_t + k_row, k_accessor, get_write_ptr(cb_k));
                        noc_async_read_barrier();
                        cb_push_back(cb_k, 1);
                    }
                }

                // V tiles: B_kv_t rows × D_t cols
                for (uint32_t v_row = 0; v_row < B_kv_t; ++v_row) {
                    for (uint32_t v_col = 0; v_col < D_t; ++v_col) {
                        cb_reserve_back(cb_v, 1);
                        noc_async_read_tile(
                            v_base + (kv_col_start + v_row) * D_t + v_col, v_accessor, get_write_ptr(cb_v));
                        noc_async_read_barrier();
                        cb_push_back(cb_v, 1);
                    }
                }

                // Mask tiles: B_q_t rows × B_kv_t cols
                if constexpr (is_causal) {
                    // Causal mask: generated on-device per (qr, kc) tile pair.
                    // Three regions:
                    //   - fully-past (kv tile end <= q tile start): all zeros
                    //   - fully-future (kv tile start >= q tile end): all -inf
                    //   - diagonal-straddling: per-element causal mask
                    for (uint32_t qr = 0; qr < B_q_t; ++qr) {
                        uint32_t q_tile_global = q_row_start + qr;
                        for (uint32_t kc = 0; kc < B_kv_t; ++kc) {
                            uint32_t kv_tile_global = kv_col_start + kc;
                            cb_reserve_back(cb_mask, 1);
                            volatile tt_l1_ptr uint16_t* tile =
                                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_mask));

                            if (kv_tile_global + 1 <= q_tile_global) {
                                // Fully-past: entire tile is in the lower triangle (attend)
                                fill_mask_tile_uniform(tile, BF16_ZERO);
                            } else if (kv_tile_global >= q_tile_global + 1) {
                                // Fully-future: entire tile is above the diagonal (mask out)
                                fill_mask_tile_uniform(tile, BF16_NEG_INF);
                            } else {
                                // Diagonal-straddling: per-element causal mask
                                generate_causal_mask_tile(tile, q_tile_global, kv_tile_global);
                            }
                            cb_push_back(cb_mask, 1);
                        }
                    }
                } else if constexpr (has_mask) {
                    // Custom additive mask: read from DRAM
                    for (uint32_t qr = 0; qr < B_q_t; ++qr) {
                        for (uint32_t kc = 0; kc < B_kv_t; ++kc) {
                            cb_reserve_back(cb_mask, 1);
                            noc_async_read_tile(mask_base + (q_row_start + qr) * S_kv_tiles + (kv_col_start + kc), mask_accessor, get_write_ptr(cb_mask));
                            noc_async_read_barrier();
                            cb_push_back(cb_mask, 1);
                        }
                    }
                }
            }
        }
    }
}
