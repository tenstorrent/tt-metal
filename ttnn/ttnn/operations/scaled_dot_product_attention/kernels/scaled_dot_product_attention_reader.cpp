// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Flash Attention reader kernel.
// Reads Q, K, V tiles from DRAM into L1 CBs.
// Prepares reduce scaler tiles and scale factor tile.
//
// When is_causal=True: generates causal (lower-triangular) mask tiles
// on-device per (Q-block, KV-block) pair instead of reading an additive
// mask tensor from DRAM.
//
// Refinement 4 — L1 budget fit:
// D-chunk loop is OUTSIDE the KV-block loop. For each D-chunk, Q and K
// are re-read from DRAM and QK^T is recomputed (weights-restreaming
// pattern). This keeps cb_o, cb_o_accum, cb_v, cb_out bounded by D_BLOCK.
// For fp32 (use_k_blocking), Q and K are pushed in k_block_dim-sized
// K-blocks so cb_q and cb_k are also constant-bounded.
//
// Work distribution: each core processes its assigned (B, H) work units
// sequentially. For GQA/MQA, maps Q-head index to KV-head index.
//
// Refinement 6 — Large sequence causal attention (S=131072):
//   Causal block skip: when is_causal, fully-future KV-blocks (all positions
//   above the causal diagonal) are skipped — no Q/K/V/mask tiles are pushed.
//   This halves the reader's work for causal attention, which was the
//   bottleneck on S=131072 (reader stuck in fill_mask_tile_uniform).
//   Also, MAX_B_KV_T increased from 4 to 8 to halve the KV-block count.
//
// Refinement 7 — Batch NoC reads for large sequences:
//   Instead of per-tile cb_reserve_back + noc_async_read_tile + barrier +
//   cb_push_back (which serializes every NoC read), batch all tile reads
//   for a block: cb_reserve_back(N) + batch reads + single barrier +
//   cb_push_back(N). This eliminates ~25M individual NoC barriers on
//   S=131072, reducing reader time from ~70s to under 5s.

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

// bf16 tile size (used for causal mask CB which is always bf16)
constexpr uint32_t BF16_TILE_BYTES = 2048;

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

    // RT args: [num_work_units, B_q_t, B_kv_t, D_t, S_q_tiles, S_kv_tiles,
    //           D_BLOCK, num_d_chunks, use_k_blocking, k_block_dim,
    //           (b,h)*num_work_units, q_addr, k_addr, v_addr, scale_bits, mask_addr]
    uint32_t rt_idx = 0;
    uint32_t num_work_units = get_arg_val<uint32_t>(rt_idx++);
    uint32_t B_q_t = get_arg_val<uint32_t>(rt_idx++);
    uint32_t B_kv_t = get_arg_val<uint32_t>(rt_idx++);
    uint32_t D_t = get_arg_val<uint32_t>(rt_idx++);
    uint32_t S_q_tiles = get_arg_val<uint32_t>(rt_idx++);
    uint32_t S_kv_tiles = get_arg_val<uint32_t>(rt_idx++);
    uint32_t D_BLOCK = get_arg_val<uint32_t>(rt_idx++);
    uint32_t num_d_chunks = get_arg_val<uint32_t>(rt_idx++);
    uint32_t use_k_blocking = get_arg_val<uint32_t>(rt_idx++);
    uint32_t k_block_dim = get_arg_val<uint32_t>(rt_idx++);

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
    constexpr auto q_args = TensorAccessorArgs<5>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();

    const auto q_accessor = TensorAccessor(q_args, q_addr);
    const auto k_accessor = TensorAccessor(k_args, k_addr);
    const auto v_accessor = TensorAccessor(v_args, v_addr);
    [[maybe_unused]] const auto mask_accessor = TensorAccessor(mask_args, mask_addr);

    // L1 page sizes for batch address advancement
    uint32_t q_tile_bytes = q_accessor.get_aligned_page_size();
    uint32_t k_tile_bytes = k_accessor.get_aligned_page_size();
    uint32_t v_tile_bytes = v_accessor.get_aligned_page_size();

    uint32_t h_q_div_h_kv = H_q / H_kv;
    uint32_t num_q_blocks = (S_q_tiles + B_q_t - 1) / B_q_t;
    uint32_t num_kv_blocks = (S_kv_tiles + B_kv_t - 1) / B_kv_t;
    uint32_t num_k_blocks = D_t / k_block_dim;

    // Refinement 7: tile counts per block (precomputed for batch reads)
    uint32_t num_q_tiles_non_kblock = B_q_t * D_t;   // Q tiles per D-chunk (non-K-blocking)
    uint32_t num_k_tiles_non_kblock = D_t * B_kv_t;  // K tiles per KV-block (non-K-blocking)
    uint32_t num_v_tiles = B_kv_t * D_BLOCK;         // V tiles per KV-block
    uint32_t num_mask_tiles = B_q_t * B_kv_t;        // mask tiles per KV-block
    // K-blocking: Q and K tiles per K-block
    uint32_t num_q_tiles_kblock = B_q_t * k_block_dim;   // Q tiles per K-block
    uint32_t num_k_tiles_kblock = k_block_dim * B_kv_t;  // K tiles per K-block

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
            uint32_t qb_end_tile = (qb + 1) * B_q_t;

            // D-chunk loop OUTSIDE KV-block loop.
            // Q and K are re-read per D-chunk (weights-restreaming).
            for (uint32_t dc = 0; dc < num_d_chunks; ++dc) {
                uint32_t d_start_v = dc * D_BLOCK;  // V column offset for this D-chunk

                for (uint32_t kvb = 0; kvb < num_kv_blocks; ++kvb) {
                    uint32_t kv_col_start = kvb * B_kv_t;

                    // Refinement 6: skip fully-future KV-blocks when is_causal.
                    if constexpr (is_causal) {
                        if (kv_col_start >= qb_end_tile) {
                            continue;
                        }
                    }

                    // --- Refinement 7: Batch NoC reads ---
                    // Instead of per-tile reserve+read+barrier+push, batch all
                    // reads for each CB block and issue a single barrier.

                    if (use_k_blocking) {
                        // K-blocking: push Q and K in k_block_dim-sized K-blocks.
                        // Q is consumed per K-block (not retained across KV-blocks).
                        for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
                            uint32_t d_start_qk = kb * k_block_dim;

                            // Batch Q tiles: B_q_t rows × k_block_dim cols
                            cb_reserve_back(cb_q, num_q_tiles_kblock);
                            {
                                uint32_t l1_addr = get_write_ptr(cb_q);
                                for (uint32_t r = 0; r < B_q_t; ++r) {
                                    for (uint32_t d = 0; d < k_block_dim; ++d) {
                                        noc_async_read_tile(
                                            q_base + (q_row_start + r) * D_t + d_start_qk + d, q_accessor, l1_addr);
                                        l1_addr += q_tile_bytes;
                                    }
                                }
                                noc_async_read_barrier();
                            }
                            cb_push_back(cb_q, num_q_tiles_kblock);

                            // Batch K tiles: k_block_dim rows × B_kv_t cols
                            cb_reserve_back(cb_k, num_k_tiles_kblock);
                            {
                                uint32_t l1_addr = get_write_ptr(cb_k);
                                for (uint32_t k_row = 0; k_row < k_block_dim; ++k_row) {
                                    for (uint32_t k_col = 0; k_col < B_kv_t; ++k_col) {
                                        noc_async_read_tile(
                                            k_base + (kv_col_start + k_col) * D_t + d_start_qk + k_row,
                                            k_accessor,
                                            l1_addr);
                                        l1_addr += k_tile_bytes;
                                    }
                                }
                                noc_async_read_barrier();
                            }
                            cb_push_back(cb_k, num_k_tiles_kblock);
                        }
                    } else {
                        // Non-K-blocking: push Q tiles once (retained across KV-blocks
                        // within this D-chunk via WaitAndRetainOnLastBlock).
                        // Only push on the first KV-block; subsequent KV-blocks reuse.
                        if (kvb == 0) {
                            // Batch Q tiles: B_q_t rows × D_t cols
                            cb_reserve_back(cb_q, num_q_tiles_non_kblock);
                            {
                                uint32_t l1_addr = get_write_ptr(cb_q);
                                for (uint32_t r = 0; r < B_q_t; ++r) {
                                    for (uint32_t d = 0; d < D_t; ++d) {
                                        noc_async_read_tile(q_base + (q_row_start + r) * D_t + d, q_accessor, l1_addr);
                                        l1_addr += q_tile_bytes;
                                    }
                                }
                                noc_async_read_barrier();
                            }
                            cb_push_back(cb_q, num_q_tiles_non_kblock);
                        }
                        // Batch K tiles: D_t rows × B_kv_t cols
                        cb_reserve_back(cb_k, num_k_tiles_non_kblock);
                        {
                            uint32_t l1_addr = get_write_ptr(cb_k);
                            for (uint32_t k_row = 0; k_row < D_t; ++k_row) {
                                for (uint32_t k_col = 0; k_col < B_kv_t; ++k_col) {
                                    noc_async_read_tile(
                                        k_base + (kv_col_start + k_col) * D_t + k_row, k_accessor, l1_addr);
                                    l1_addr += k_tile_bytes;
                                }
                            }
                            noc_async_read_barrier();
                        }
                        cb_push_back(cb_k, num_k_tiles_non_kblock);
                    }

                    // Batch V tiles: B_kv_t rows × D_BLOCK cols for this D-chunk
                    cb_reserve_back(cb_v, num_v_tiles);
                    {
                        uint32_t l1_addr = get_write_ptr(cb_v);
                        for (uint32_t v_row = 0; v_row < B_kv_t; ++v_row) {
                            for (uint32_t v_col = 0; v_col < D_BLOCK; ++v_col) {
                                noc_async_read_tile(
                                    v_base + (kv_col_start + v_row) * D_t + d_start_v + v_col, v_accessor, l1_addr);
                                l1_addr += v_tile_bytes;
                            }
                        }
                        noc_async_read_barrier();
                    }
                    cb_push_back(cb_v, num_v_tiles);

                    // Batch mask tiles: B_q_t rows × B_kv_t cols
                    if constexpr (is_causal) {
                        // Causal mask: generate on-device, no NoC reads needed.
                        // Batch all tiles: reserve, fill all, push all.
                        // Mask CB is always bf16 for causal path.
                        cb_reserve_back(cb_mask, num_mask_tiles);
                        {
                            uint32_t l1_addr = get_write_ptr(cb_mask);
                            for (uint32_t qr = 0; qr < B_q_t; ++qr) {
                                uint32_t q_tile_global = q_row_start + qr;
                                for (uint32_t kc = 0; kc < B_kv_t; ++kc) {
                                    uint32_t kv_tile_global = kv_col_start + kc;
                                    volatile tt_l1_ptr uint16_t* tile =
                                        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr);

                                    if (kv_tile_global + 1 <= q_tile_global) {
                                        fill_mask_tile_uniform(tile, BF16_ZERO);
                                    } else if (kv_tile_global >= q_tile_global + 1) {
                                        fill_mask_tile_uniform(tile, BF16_NEG_INF);
                                    } else {
                                        generate_causal_mask_tile(tile, q_tile_global, kv_tile_global);
                                    }
                                    l1_addr += BF16_TILE_BYTES;
                                }
                            }
                        }
                        cb_push_back(cb_mask, num_mask_tiles);
                    } else if constexpr (has_mask) {
                        // Custom mask: batch read from DRAM
                        cb_reserve_back(cb_mask, num_mask_tiles);
                        {
                            uint32_t l1_addr = get_write_ptr(cb_mask);
                            // mask CB is input dtype (not causal), so use q_tile_bytes
                            // (mask accessor has same page size as input)
                            uint32_t mask_tile_bytes = mask_accessor.get_aligned_page_size();
                            for (uint32_t qr = 0; qr < B_q_t; ++qr) {
                                for (uint32_t kc = 0; kc < B_kv_t; ++kc) {
                                    noc_async_read_tile(
                                        mask_base + (q_row_start + qr) * S_kv_tiles + (kv_col_start + kc),
                                        mask_accessor,
                                        l1_addr);
                                    l1_addr += mask_tile_bytes;
                                }
                            }
                            noc_async_read_barrier();
                        }
                        cb_push_back(cb_mask, num_mask_tiles);
                    }
                }
            }
        }
    }
}
